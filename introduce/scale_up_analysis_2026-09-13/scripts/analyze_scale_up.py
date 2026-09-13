#!/usr/bin/env python3
"""Recompute descriptive findings from audited local data; never fit a scaling law.

Python standard library suffices for JSON/CSV. Pass --plots with matplotlib installed
to regenerate publication-style PNG/SVG figures. No training or network calls.
"""
import argparse
import csv
import hashlib
import json
from decimal import Decimal as D
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
DATA = OUT / 'data'
BASE = 'base(bn-senet-dcnm)'
SMALL = 'bn_rankmixer_v6_e2_small'
S3 = SMALL + '_3'
E2 = 'bn_rankmixer_v6_e2'
MAT = 'cvr-senet-mature-rankmixer-v1'
NAMES = {BASE: 'Base', SMALL: 'Small (L2)', S3: 'Small-3 (L3)', E2: 'E2 (D512)', MAT: 'Mature-v1'}


def read_csv(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (DATA / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def average(values):
    return sum(values, D(0)) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    rows = read_csv('experiment_results.csv')
    capacity = {r['model']: r for r in read_csv('model_capacity.csv')}
    index = {(r['model_id'], r['test_date']): r for r in rows}
    assert len(index) == len(rows) == 97
    for r in rows:
        assert D(r['auc']) - D(r['base_auc']) == D(r['auc_gap_vs_base'])
    # Independent verification of every saved pair and its direction: right-left.
    for pair in read_csv('paired_comparisons.csv'):
        dates = pair['dates'].split(';')
        delta = [D(index[pair['right_model'], t]['auc']) - D(index[pair['left_model'], t]['auc']) for t in dates]
        assert abs(average(delta) - D(pair['mean_delta'])) < D('1e-12')
        assert len(delta) == int(pair['n_dates'])
    dates = [f'2026-08-{i}' for i in range(16, 21)]
    params = {BASE: 90341785, **{m: int(capacity[c]['dense_params_static_fact']) for m, c in
              [(SMALL, 'v6_e2_small'), (S3, 'v6_e2_small_3'), (E2, 'v6_e2'), (MAT, 'mature_v1')]}}
    flops = {BASE: 180923051, **{m: int(capacity[c]['flops_per_sample_estimate']) for m, c in
              [(SMALL, 'v6_e2_small'), (S3, 'v6_e2_small_3'), (E2, 'v6_e2'), (MAT, 'mature_v1')]}}
    window = []
    for model in NAMES:
        records = [index[model, t] for t in dates]
        window.append(dict(model_id=model, label=NAMES[model], start=dates[0], end=dates[-1], n_dates=len(dates),
                           mean_auc=average([D(r['auc']) for r in records]),
                           mean_gap=average([D(r['auc_gap_vs_base']) for r in records]),
                           mean_abs_copc_error=average([abs(D(r['copc'])-1) for r in records]),
                           dense_params_current_static=params[model], forward_flops_static_estimate=flops[model],
                           historical_config_binding='unverified', auc_cells=';'.join(r['auc_cell'] for r in records)))
    write_csv('matched_five_day_summary.csv', window)
    daily = [dict(test_date=t, s3_minus_small=D(index[S3,t]['auc'])-D(index[SMALL,t]['auc']),
                  s3_minus_e2=D(index[S3,t]['auc'])-D(index[E2,t]['auc']),
                  s3_minus_base=D(index[S3,t]['auc_gap_vs_base']),
                  mature_minus_s3=D(index[MAT,t]['auc'])-D(index[S3,t]['auc'])) for t in dates]
    write_csv('scale_up_daily_contrasts.csv', daily)
    light_gain = D(index[SMALL+'_2',dates[0]]['auc'])-D(index[SMALL+'_1',dates[0]]['auc'])
    result = {
        'units': 'AUC absolute difference; one basis point=0.0001; no CI from summary rows',
        'small3_minus_small_five_day': average([r['s3_minus_small'] for r in daily]),
        'small3_minus_e2_five_day': average([r['s3_minus_e2'] for r in daily]),
        'small3_minus_base_five_day': average([r['s3_minus_base'] for r in daily]),
        'mature_minus_small3_five_day': average([r['mature_minus_s3'] for r in daily]),
        'full_depth_gain_first_day': daily[0]['s3_minus_small'],
        'light_depth_gain_first_day': light_gain,
        'full_minus_light_depth_interaction_first_day': daily[0]['s3_minus_small']-light_gain,
        'small3_param_growth_vs_small_pct': 100*D(params[S3]-params[SMALL])/params[SMALL],
        'small3_param_saving_vs_e2_pct': 100*D(params[E2]-params[S3])/params[E2],
        'small_param_saving_vs_e2_pct': 100*D(params[E2]-params[SMALL])/params[E2],
        'small3_param_growth_vs_base_pct': 100*D(params[S3]-params[BASE])/params[BASE],
        'small3_flops_saving_vs_e2_pct': 100*D(flops[E2]-flops[S3])/flops[E2],
        'small3_vs_e2_gap_closed_fraction': average([r['s3_minus_e2'] for r in daily]) / -average([D(index[E2,t]['auc_gap_vs_base']) for t in dates]),
        'cannot_claim': ['universal monotonic scaling','causal depth-over-width at equal compute','statistical significance','throughput speedup','data scaling exponent']
    }
    assert result['small3_minus_small_five_day'] == D('0.0001892')
    assert result['full_minus_light_depth_interaction_first_day'] == D('-0.000127')
    (DATA/'headline_findings.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
    # Independent parameter-budget checks, including prospective, unrun points.
    proposals = []
    for d,l,m in [(256,2,704),(256,3,704),(352,2,704),(256,4,704),(256,2,1056)]:
        p = 5295973+d*d+107589*d+64*l*(3*d*m+2*m+2*d)
        proposals.append(dict(D=d,L=l,M=m,dense_params_static_budget=p,status='proposal_not_run'))
    recorded = json.loads((DATA/'capacity_proposals.json').read_text())
    for r in proposals:
        assert recorded[f'D{r["D"]}_L{r["L"]}_M{r["M"]}']['total'] == r['dense_params_static_budget']
    write_csv('proposed_capacity_grid.csv', proposals)
    if args.plots:
        make_plots(index, dates, daily, window, capacity)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


def make_plots(index, dates, daily, window, capacity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white'})
    assets = OUT/'assets'
    assets.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(assets/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(assets/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for col,label,color in [('s3_minus_small','Small-3 minus Small','#16697a'),('s3_minus_e2','Small-3 minus E2','#cb6a29')]:
        ax[0].plot([t[5:] for t in dates],[float(r[col])*10000 for r in daily],marker='o',label=label,color=color,lw=2)
    ax[0].axhline(0,color='#777777',lw=.8)
    ax[0].set(title='Local depth evidence | matched five days',xlabel='Test date (2026)',ylabel='AUC difference (basis points; 1 bp = 0.0001)')
    ax[0].legend(loc='upper left',frameon=False)
    for models,label,color in [([SMALL,S3],'Full terminal','#16697a'),([SMALL+'_1',SMALL+'_2'],'Light terminal','#cb6a29')]:
        yy=[float(index[m,dates[0]]['auc_gap_vs_base'])*10000 for m in models]
        ax[1].plot([2,3],yy,marker='o',lw=2,color=color,label=label)
        for x,y in zip([2,3],yy): ax[1].annotate(f'{y:+.2f}',(x,y),xytext=(0,8),textcoords='offset points',ha='center')
    ax[1].set(title='Depth x terminal | first day only',xticks=[2,3],xlim=(1.85,3.3),ylim=(-9.5,-2.5),xlabel='Number of dual-FFN blocks',ylabel='AUC minus same-day Base (bp)')
    ax[1].legend(frameon=False,loc='upper left')
    fig.suptitle('Observed local gains, with no confidence intervals or extrapolated trend',fontsize=13)
    save(fig,'01_depth_and_terminal')
    fig, ax = plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    colors={BASE:'#3d4856',SMALL:'#83a65d',S3:'#16697a',E2:'#cb6a29',MAT:'#8064a2'}
    offsets={BASE:(5,9),SMALL:(5,-17),S3:(6,4),E2:(-85,6),MAT:(5,9)}
    for r in window:
        model=r['model_id']; x=r['dense_params_current_static']/1e6; y=float(r['mean_gap'])*1e4
        ax[0].scatter(x,y,s=70,color=colors[model],zorder=3)
        ax[0].annotate(NAMES[model],(x,y),xytext=offsets[model],textcoords='offset points')
    ax[0].axhline(0,color='#777777',lw=.8)
    ax[0].set(title='Capacity and ranking | Aug 16-20',xlabel='Current-source Dense parameters (millions)',ylabel='Mean same-day AUC gap vs Base (bp)',xlim=(80,215),ylim=(-5.7,1.2))
    names=[r['label'] for r in window]
    ax[1].barh(names,[100*float(r['mean_abs_copc_error']) for r in window],color=[colors[r['model_id']] for r in window])
    ax[1].invert_yaxis()
    ax[1].set(title='Calibration-scale tradeoff | same five days',xlabel='Mean daily |COPC - 1| (%)')
    ax[1].set_xlim(0,1.6)
    for i,r in enumerate(window): ax[1].text(100*float(r['mean_abs_copc_error'])+.025,i,f'{100*float(r["mean_abs_copc_error"]):.3f}%',va='center')
    fig.suptitle('Different architectures; static capacity is not runtime cost or a scaling-law fit',fontsize=13)
    save(fig,'02_capacity_and_copc')
    # All August first-day candidates: no invented capacity for UniMixer or conflicting mature-v5.
    first=sorted([r for (m,t),r in index.items() if t==dates[0]],key=lambda r:float(r['auc']))
    labels=[]; costs=[]
    for r in first:
        model=r['model_id']
        label=NAMES.get(model,model.replace('bn_rankmixer_','').replace('cvr-senet-mature-rankmixer-','Mature-'))
        label=label.replace('v6_e2_small_1','Small-1').replace('v6_e2_small_2','Small-2').replace('v6_e3','E3').replace('v6_e4','E4')
        labels.append(label)
        key=model.replace('bn_rankmixer_','').replace('cvr-senet-mature-rankmixer-','mature_')
        costs.append('90.342M' if model==BASE else 'capacity unresolved' if key=='mature_v5' else
                     f'{int(capacity[key]["dense_params_static_fact"])/1e6:.3f}M' if key in capacity else 'not audited')
    fig, ax = plt.subplots(figsize=(10.5,7.5),layout='constrained')
    ys=list(range(len(first))); vals=[float(r['auc_gap_vs_base'])*10000 for r in first]
    ax.barh(ys,vals,color=[colors.get(r['model_id'],'#b3bdc8') for r in first])
    ax.set_yticks(ys,labels)
    ax.set_xlim(-33,8)
    for y,val,cost in zip(ys,vals,costs):
        ax.text(val-.3,y,f'{val:+.2f}',ha='right',va='center',fontsize=9)
        ax.text(.8,y,cost,ha='left',va='center',fontsize=9)
    ax.axvline(0,color='#777777',lw=.8)
    ax.set(title='All observed August first-day results | 2026-08-16',xlabel='AUC minus same-day Base (bp); right labels: current static Dense capacity')
    save(fig,'03_first_day_coverage')


if __name__=='__main__':
    main()
