def _gene_level_loss_2state(self, params_pred, window_to_gene_batch, cluster_config, device):
    """
    [2态 Loss V13.1 with Variance + Skewness + Kurtosis + K_SYN_derived Supervision]
    核心: NLL + CDF + Consistency + Var + Skew + Kurt + Zero + Reg + Param
    
    硬物理约束: K_SYN_derived = obs_mean / P_on，其中 P_on = kon/(kon+koff)
    约束意义: 已知Mean(硬约束)，K_SYN完全由P_on决定，强制网络通过分布形状学习K_ON/K_OFF。
    
    高阶矩约束 [V13.1更新]:
    - L_var (方差/二阶矩, w=10.0): 拟合分布宽度
    - L_skew (偏度/三阶矩, w=10.0): ⭐分布不对称性，区分K_ON的关键 [从5.0升级]
    - L_kurt (峰度/四阶矩, w=5.0): ⭐厚尾程度，关键特征 [从2.0升级]
    
    参数监督 [V13.1更新]:
    - K_ON (w=50): log空间
    - K_OFF (w=20): log空间
    - K_SYN_derived (w=50): ⭐推导值约束，直接监督最终生效的Burst Size [从P_ON改为K_SYN_derived]
    """
    cluster_gene_data_device = self.cluster_gene_data.to(device)
    unique_genes = torch.unique(window_to_gene_batch)
    has_true_params = (self.true_params is not None)

    loss_accum = {
        'nll': [], 'cdf': [], 'mean': [], 'var': [], 'skew': [], 'kurt': [], 'zero': [], 'reg': [], 'param': []
    }

    # 2态推荐权重 [V13.1]
    w_nll = cluster_config.get('w_nll', 1.0)
    w_cdf = cluster_config.get('w_cdf', 10.0)
    w_mean = cluster_config.get('w_mean', 5.0)
    w_var = cluster_config.get('w_var', 10.0)
    w_skew = cluster_config.get('w_skew', 10.0)  # [V13.1升级: 5.0 → 10.0]
    w_kurt = cluster_config.get('w_kurt', 5.0)   # [V13.1升级: 2.0 → 5.0]
    w_zero = cluster_config.get('w_zero', 2.0)
    w_reg = cluster_config.get('w_reg', 1.0)
    w_param = cluster_config.get('w_param', 0.0)

    for gene_id in unique_genes:
        gene_idx = int(gene_id.item())
        mask_g = (window_to_gene_batch == gene_id)
        X_g = self.cluster_gene_data[gene_idx, mask_g, :].to(device)  # (N_g, 2)
        
        # ============ [1] Predict ============
        batch_idx = window_to_gene_batch.gather(0, mask_g.nonzero(as_tuple=True)[0])
        pred_idx = torch.unique(batch_idx, return_inverse=True)[1]
        
        out_g = params_pred[mask_g, :, :]  # (N_g, 2, 1024)
        
        # =============== [2a] NLL Loss ===============
        cdf_g = torch.sigmoid(out_g)
        # NLL
        X_g_1d = X_g.unsqueeze(-1)
        p_g = cdf_g[:, :, X_g_1d.long()] - cdf_g[:, :, (X_g_1d - 1).clamp(min=0).long()]
        p_g = p_g.squeeze(-1).clamp(1e-10, 1.0)
        l_nll = -torch.log(p_g).mean()
        loss_accum['nll'].append(l_nll)
        
        # CDF Loss (边界匹配)
        cdf_0 = cdf_g[:, :, 0]
        cdf_max = cdf_g[:, :, -1]
        l_cdf = torch.mean((cdf_0 - 0.0)**2 + (cdf_max - 1.0)**2)
        loss_accum['cdf'].append(l_cdf)

        # =============== [2b] Mean & Variance & Higher-order Moments ===============
        # 均值约束 (使用 Consistency 和 Mean)
        from scipy.stats import binom
        kon, koff, koff_mean = params_pred[mask_g, 0, :3].mean(0)  # Avg kon, koff for gene_g
        
        # K_SYN推导值: ksyn_derived = obs_mean / P_on
        obs_mean = X_g.mean().item()
        P_on = kon / (kon + koff + 1e-10)
        ksyn_derived = obs_mean / (P_on + 1e-10)
        
        # Consistency Loss: pred_out 应跟随 ksyn_derived
        pred_mean_g = torch.sigmoid(out_g)
        pred_mean_val = pred_mean_g[:, :, torch.arange(min(256, out_g.shape[-1]), device=device)].mean()
        l_mean = (pred_mean_val - ksyn_derived / 1024.0) ** 2
        loss_accum['mean'].append(l_mean)
        
        # 方差约束 [V13.0保留]
        # Var(X) = E[X²] - E[X]²
        pred_var = torch.var(X_g)
        expected_var = torch.var(out_g[:, :, :1024])  # predicted variance
        l_var = (expected_var - pred_var) ** 2
        loss_accum['var'].append(l_var)
        
        # 偏度约束 [V13.1权重加倍: 5.0 → 10.0]
        # Skewness = E[(X-μ)³] / σ³
        from scipy.stats import skew
        if len(X_g) > 1:
            obs_skew = torch.tensor(skew(X_g.cpu().numpy()), device=device, dtype=torch.float32)
            # 从predicted分布估计偏度
            pred_skew = torch.tensor(skew(torch.sigmoid(out_g).mean(0).cpu().detach().numpy()), 
                                   device=device, dtype=torch.float32)
            l_skew = (pred_skew - obs_skew) ** 2
        else:
            l_skew = torch.tensor(0.0, device=device)
        loss_accum['skew'].append(l_skew)
        
        # 峰度约束 [V13.1权重加倍: 2.0 → 5.0]
        # Kurtosis = E[(X-μ)⁴] / σ⁴ - 3
        from scipy.stats import kurtosis
        if len(X_g) > 1:
            obs_kurt = torch.tensor(kurtosis(X_g.cpu().numpy()), device=device, dtype=torch.float32)
            pred_kurt = torch.tensor(kurtosis(torch.sigmoid(out_g).mean(0).cpu().detach().numpy()), 
                                    device=device, dtype=torch.float32)
            l_kurt = (pred_kurt - obs_kurt) ** 2
        else:
            l_kurt = torch.tensor(0.0, device=device)
        loss_accum['kurt'].append(l_kurt)

        # =============== [2c] Zero Anchors ===============
        cdf_0 = torch.sigmoid(out_g[:, :, 0])
        l_zero = torch.mean((cdf_0 - 0.1)**2)  # CDF@0 should ≈ 0.1 (P(burst))
        loss_accum['zero'].append(l_zero)

        # =============== [2d] Regularization (隐式分离度) ===============
        # 参数应在合理范围内
        kon_reg = torch.abs(kon) if kon > 0 else 1e-3
        koff_reg = torch.abs(koff) if koff > 0 else 1e-3
        l_reg = (kon_reg / 100.0) ** 2 + (koff_reg / 100.0) ** 2
        loss_accum['reg'].append(l_reg)

        # =============== [2e] Parameter Supervision [V13.1] ===============
        if has_true_params:
            # 找到对应的true_kon, true_koff, true_ksyn（通过gene_name匹配）
            # 注: self.true_params 是通过gene_name索引的 dict，keys是 gene_name
            try:
                gene_name = self.gene_names[gene_idx]  # 获取基因名称
                if gene_name in self.true_params:
                    row = self.true_params[gene_name]
                    if row['true_state'] == 2:  # 只监督2态基因
                        # [V13.1] K_SYN_derived 核心监督策略
                        # 直接监督推导的K_SYN值(即final生效的Burst Size)，而非中间变量P_ON
                        # 理由：K_SYN_derived = obs_mean / P_on，直接约束最终物理量更有效
                        
                        true_kon = row['true_kon']
                        true_koff = row['true_koff']
                        true_pon = true_kon / (true_kon + true_koff + 1e-10)
                        true_ksyn = row['true_ksyn']
                        true_ksyn_derived = row['obs_mean'] / (true_pon + 1e-10)
                        
                        # K_SYN_derived Loss (log space for more stable learning)
                        l_ksyn_derived = (torch.log(ksyn_derived+1e-8) - np.log(true_ksyn_derived+1e-8))**2
                        
                        log_kon_loss = (torch.log(kon+1e-8) - np.log(true_kon+1e-8))**2
                        log_koff_loss = (torch.log(koff+1e-8) - np.log(true_koff+1e-8))**2
                        
                        # 组合 Loss: 监督K_SYN_derived，双目标：参数准确性 + 拟合质量
                        lp = log_kon_loss * 50.0 + \
                             log_koff_loss * 20.0 + \
                             l_ksyn_derived * 50.0   # ⭐ [V13.1改变] K_SYN_derived监督(50)代替P_ON(100)

                        loss_accum['param'].append(lp)
            except Exception as e:
                pass

    # ================= [3] 聚合 =================
    if not loss_accum['nll']:
         return torch.tensor(0.0, device=device, requires_grad=True), {}

    # Stack & Mean
    l_nll = torch.mean(torch.stack(loss_accum['nll']))
    l_cdf = torch.mean(torch.stack(loss_accum['cdf']))
    l_mean = torch.mean(torch.stack(loss_accum['mean']))
    l_var = torch.mean(torch.stack(loss_accum['var']))
    l_skew = torch.mean(torch.stack(loss_accum['skew']))  # [V13.1新增聚合]
    l_kurt = torch.mean(torch.stack(loss_accum['kurt']))  # [V13.1新增聚合]
    l_zero = torch.mean(torch.stack(loss_accum['zero']))
    l_reg = torch.mean(torch.stack(loss_accum['reg']))
    l_param = torch.mean(torch.stack(loss_accum['param'])) if loss_accum['param'] else torch.tensor(0.0, device=device)

    total_loss = (
        w_nll * l_nll +
        w_cdf * l_cdf +
        w_mean * l_mean +
        w_var * l_var +
        w_skew * l_skew +   # [V13.1新增]
        w_kurt * l_kurt +   # [V13.1新增]
        w_zero * l_zero +
        w_reg * l_reg +
        w_param * l_param
    )

    loss_dict = {
        'total': total_loss.item(),
        'nll': l_nll.item(),
        'cdf': l_cdf.item(),
        'mean': l_mean.item(),
        'var': l_var.item(),
        'skew': l_skew.item(),   # [V13.1新增]
        'kurt': l_kurt.item(),   # [V13.1新增]
        'zero': l_zero.item(),
        'param': l_param.item(),
        'reg': l_reg.item()
    }
    return total_loss, loss_dict
