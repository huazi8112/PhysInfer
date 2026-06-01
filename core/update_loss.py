import sys

with open('c:/Users/zengxiaodong/Desktop/华子哥的任务/111/222/方案B_状态数驱动架构.py', 'r', encoding='utf-8') as f:
    content = f.read()

start = content.find('class PhysicsMomentLoss(')
end = content.find('\nclass ResultVisualizer', start)

new_class = '''class PhysicsMomentLoss(nn.Module):
    \"\"\"矩匹配损失：采用严格解析的 Telegraph 模型阶乘矩 (Factorial Moments) 进行拟合。
    这能够完全闭合地解决参数不可识别性和模式坍缩问题\"\"\"
    def __init__(self, eps=1e-6, w_m1=10.0, w_m2=5.0, w_m3=2.0):
        super().__init__()
        self.eps = eps
        self.w_m1 = w_m1
        self.w_m2 = w_m2
        self.w_m3 = w_m3

    def forward(self, model_out, obs_counts):
        batch_size = len(obs_counts)
        total_loss = 0.0
        loss_dict = {'m1': 0.0, 'm2': 0.0, 'm3': 0.0, 'total': 0.0, 'total_n': 0}

        kon_pred = model_out.get('k_on', None)
        koff_pred = model_out.get('k_off', None)
        ksyn_pred = model_out.get('k_syn', None)
        BF_pred = model_out.get('BF', None)
        BS_pred = model_out.get('BS', None)

        for i in range(batch_size):
            data = obs_counts[i].float()
            if data.numel() == 0:
                continue
                
            obs_m1 = torch.mean(data)
            obs_m2 = torch.mean(data * (data - 1.0))
            obs_m3 = torch.mean(data * (data - 1.0) * (data - 2.0))
            
            obs_m1_safe = torch.clamp(obs_m1, min=1e-4)
            obs_m2_safe = torch.clamp(obs_m2, min=1e-4)
            obs_m3_safe = torch.clamp(obs_m3, min=1e-4)

            if kon_pred is not None and koff_pred is not None and ksyn_pred is not None:
                kon = torch.clamp(kon_pred[i], min=1e-3, max=1e4)
                koff = torch.clamp(koff_pred[i], min=1e-3, max=1e4)
                ksyn = torch.clamp(ksyn_pred[i], min=1e-3, max=1e4)
            else:
                kon = torch.clamp(BF_pred[i], min=1e-3, max=1e4)
                koff = torch.clamp(torch.tensor(1.0, device=data.device), min=1e-3, max=1e4)
                ksyn = torch.clamp(BS_pred[i] * koff, min=1e-3, max=1e4)

            sum_k = kon + koff
            pred_m1 = ksyn * (kon / sum_k)
            pred_m2 = pred_m1 * ksyn * ((kon + 1.0) / (sum_k + 1.0))
            pred_m3 = pred_m2 * ksyn * ((kon + 2.0) / (sum_k + 2.0))

            import torch.nn.functional as F
            loss_m1 = F.l1_loss(torch.log(torch.clamp(pred_m1, min=1e-4)), torch.log(obs_m1_safe))
            loss_m2 = F.l1_loss(torch.log(torch.clamp(pred_m2, min=1e-4)), torch.log(obs_m2_safe))
            loss_m3 = F.l1_loss(torch.log(torch.clamp(pred_m3, min=1e-4)), torch.log(obs_m3_safe))
            
            reg_penalty = 1e-4 * (kon**2 + koff**2 + (ksyn/100)**2)

            sample_loss = self.w_m1 * loss_m1 + self.w_m2 * loss_m2 + self.w_m3 * loss_m3 + reg_penalty

            total_loss = total_loss + sample_loss

            loss_dict['m1'] += loss_m1.item()
            loss_dict['m2'] += loss_m2.item()
            loss_dict['m3'] += loss_m3.item()
            loss_dict['total_n'] += 1

        if loss_dict['total_n'] == 0:
            return torch.tensor(0.0, requires_grad=True), loss_dict

        avg_loss = total_loss / loss_dict['total_n']
        loss_dict['m1'] /= loss_dict['total_n']
        loss_dict['m2'] /= loss_dict['total_n']
        loss_dict['m3'] /= loss_dict['total_n']
        loss_dict['total'] = avg_loss.item()
        return avg_loss, loss_dict
'''

if start != -1 and end != -1:
    updated_content = content[:start] + new_class + content[end:]
    with open('c:/Users/zengxiaodong/Desktop/华子哥的任务/111/222/方案B_状态数驱动架构.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    print('SUCCESSFULLY UPDATED LOSS FILE')
else:
    print('FAILED TO FIND BLOCK')