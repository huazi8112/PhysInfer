import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "Arial"
})

def compute_3state_effective(k1_p, k1_m, k2_p, k2_m, k_ini):
    k1_p = np.clip(k1_p, 1e-8, 1e6)
    k1_m = np.clip(k1_m, 1e-8, 1e6)
    k2_p = np.clip(k2_p, 1e-8, 1e6)
    k2_m = np.clip(k2_m, 1e-8, 1e6)
    k_ini = np.clip(k_ini, 1e-8, 1e9)
    bs = k_ini / (k2_m + 1e-12)
    w_on = 1.0
    w_active_off = k2_m / (k2_p + 1e-12)
    w_deep_off = (k2_m * k1_m) / ((k2_p * k1_p) + 1e-12)
    Z = w_on + w_active_off + w_deep_off
    p_on = w_on / (Z + 1e-12)
    mean = k_ini * p_on
    bf = mean / (bs + 1e-12)
    return bf, bs, mean


def compute_2state_effective(kon, koff, ksyn):
    kon = np.clip(kon, 1e-8, 1e6)
    koff = np.clip(koff, 1e-8, 1e6)
    ksyn = np.clip(ksyn, 1e-8, 1e9)
    bs = ksyn / (koff + 1e-12)
    bf = (kon * koff) / (kon + koff + 1e-12)
    mean = ksyn * (kon / (kon + koff + 1e-12))
    return bf, bs, mean


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    src = 'plan_b_state_driven_results.csv'
    if not os.path.exists(src):
        print('MISSING', src)
        return
    df = pd.read_csv(src, encoding='utf-8-sig')

    bf_list = []
    bs_list = []
    mean_list = []
    topo = []

    for _, row in df.iterrows():
        dom = row.get('dominant_state')
        try:
            # Recognize 3-state either by dominant_state==3 or by presence of k01/k10/k12/k21
            has_kcols = any(pd.notna(row.get(c)) for c in ['k01', 'k10', 'k12', 'k21'])
            if has_kcols or (pd.notna(dom) and float(dom) == 3.0):
                # prefer precomputed BF/BS if available in results
                if pd.notna(row.get('BF')) and pd.notna(row.get('BS')):
                    bf = row.get('BF')
                    bs = row.get('BS')
                    mean = row.get('actual_mean', np.nan)
                    topo.append('3-State (Refractory)')
                else:
                    k1_p = row.get('k1_p', np.nan)
                    k1_m = row.get('k1_m', np.nan)
                    k2_p = row.get('k2_p', np.nan)
                    k2_m = row.get('k2_m', np.nan)
                    # try several possible synthesis-rate column names
                    k_ini = row.get('k_ini', row.get('ksyn2', row.get('ksyn', row.get('ksyn_raw', np.nan))))
                    # fallback to alternate column names used in results
                    if pd.isna(k1_p):
                        k1_p = row.get('k01', k1_p)
                    if pd.isna(k1_m):
                        k1_m = row.get('k10', k1_m)
                    if pd.isna(k2_p):
                        k2_p = row.get('k12', k2_p)
                    if pd.isna(k2_m):
                        k2_m = row.get('k21', k2_m)
                    if pd.isna(k_ini):
                        k_ini = row.get('ksyn', row.get('ksyn_raw', k_ini))
                    bf, bs, mean = compute_3state_effective(float(k1_p), float(k1_m), float(k2_p), float(k2_m), float(k_ini))
                    topo.append('3-State (Refractory)')
            else:
                if (pd.notna(row.get('burst_frequency')) and pd.notna(row.get('burst_size'))):
                    bf = row.get('burst_frequency')
                    bs = row.get('burst_size')
                    mean = row.get('actual_mean', np.nan)
                else:
                    kon = row.get('kon', row.get('k_on', np.nan))
                    koff = row.get('koff', row.get('k_off', np.nan))
                    ksyn = row.get('ksyn', row.get('ksyn_raw', row.get('ksyn2', np.nan)))
                    bf, bs, mean = compute_2state_effective(float(kon), float(koff), float(ksyn))
                topo.append('2-State (Telegraph)')
        except Exception:
            bf, bs, mean = (np.nan, np.nan, np.nan)
            topo.append('unknown')
        bf_list.append(bf)
        bs_list.append(bs)
        mean_list.append(mean)

    # debug counts
    from collections import Counter
    topo_counter = Counter(topo)
    print('TOPO_COUNTER_RAW:', topo_counter)

    plot_df = pd.DataFrame({'BF': bf_list, 'BS': bs_list, 'Mean_Expression': mean_list, 'Topology': topo})
    # save raw (before dropping/thresholding) for debug
    raw_csv = 'Fig3B_plot_data_planb_raw.csv'
    plot_df.to_csv(raw_csv, index=False, encoding='utf-8-sig')
    print('WROTE_RAW_CSV', raw_csv)
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['BF', 'BS'])
    plot_df = plot_df[(plot_df['BF'] > 0) & (plot_df['BS'] > 0)]

    csv_out = 'Fig3B_plot_data_planb.csv'
    plot_df.to_csv(csv_out, index=False, encoding='utf-8-sig')
    print('WROTE_CSV', csv_out)

    fig, ax = plt.subplots(figsize=(7,6))
    colors = {'2-State (Telegraph)': '#3498db', '3-State (Refractory)': '#e74c3c', 'unknown':'#777777'}
    markers = {'2-State (Telegraph)': 'o', '3-State (Refractory)': '^', 'unknown': 'x'}
    sizes = {'2-State (Telegraph)': 18, '3-State (Refractory)': 60, 'unknown': 24}
    for name, group in plot_df.groupby('Topology'):
        ax.scatter(group['BF'], group['BS'], s=sizes.get(name,20), alpha=0.7, label=name,
                   c=colors.get(name, '#999999'), marker=markers.get(name, 'o'), edgecolors='k', linewidths=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    log_bf = np.log10(plot_df['BF'])
    log_bs = np.log10(plot_df['BS'])
    pad = 0.25
    x_min = 10 ** (np.percentile(log_bf, 1) - pad)
    x_max = 10 ** (np.percentile(log_bf, 99) + pad)
    y_min = 10 ** (np.percentile(log_bs, 1) - pad)
    y_max = 10 ** (np.percentile(log_bs, 99) + pad)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Burst Frequency (BF)')
    ax.set_ylabel('Burst Size (BS)')
    # Pearson via numpy
    r_val = np.corrcoef(np.log10(plot_df['BF']), np.log10(plot_df['BS']))[0,1]
    ax.text(0.05, 0.95, f'Global Pearson r = {r_val:.2f}', transform=ax.transAxes, fontsize=10, va='top')
    ax.legend(frameon=True, loc='lower right')
    
    pdf_out = 'Fig3B_Kinetic_Manifold_planb_centered_editable.pdf'
    png_out = 'Fig3B_Kinetic_Manifold_planb_centered_editable.png'
    svg_out = 'Fig3B_Kinetic_Manifold_planb_centered_editable.svg'
    
    fig.tight_layout()
    fig.savefig(pdf_out, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(png_out, format='png', dpi=300, bbox_inches='tight')
    fig.savefig(svg_out, format='svg', bbox_inches='tight')
    
    plt.close(fig)
    print('WROTE FILES:', pdf_out, png_out, svg_out)

    # list outputs
    outs = [p for p in os.listdir(here) if p.startswith('Fig3B')]
    print('OUT_LIST:', outs)


if __name__ == '__main__':
    main()
