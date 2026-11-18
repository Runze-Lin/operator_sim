"""Comparison on one common qualified PR holdout; all durations in µs."""
from statistics import mean


def comparison(rows):
    ca = [r['pr_ca_us'] for r in rows]
    fields = ('pr_us', 'dt_us', 'prior_pr_us')
    result = {}
    for key in fields:
        if not all(r.get(key) is not None for r in rows):
            continue
        errors = [r[key] - ref for r, ref in zip(rows,ca)]
        result[key] = dict(wape=sum(map(abs,errors))/sum(ca),
                           max_ape=max(abs(d)/t for d,t in zip(errors,ca)),
                           mean_signed_us=mean(errors))
    wins = ties = 0
    for r in rows:
        difference = abs(r['dt_us'] - r['pr_ca_us']) - abs(r['pr_us'] - r['pr_ca_us'])
        wins += difference > 1e-9
        ties += abs(difference) <= 1e-9
    result.update(n=len(rows), wins=wins,ties=ties,losses=len(rows)-wins-ties)
    paired = [r for r in rows if r.get('dt_ca_us') is not None]
    if paired:
        result['reference_error_mean_us'] = mean(r['dt_us']-r['dt_ca_us'] for r in paired)
        result['transfer_error_mean_us'] = mean(
            (r['pr_us']-r['dt_us'])-(r['pr_ca_us']-r['dt_ca_us']) for r in paired)
        result['paired_count'] = len(paired)
    return result


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,3,figsize=(13,5.9),layout='constrained')
    for ax, op in zip(axes, ('quant','matmul','fa')):
        cases = sorted((r for r in rows if r['op']==op), key=lambda r:r['pr_ca_us'])
        for i,r in enumerate(cases):
            before = 100*(r['dt_us']/r['pr_ca_us']-1)
            after = 100*(r['pr_us']/r['pr_ca_us']-1)
            ax.plot([before,after],[i,i],color='#bac4ca',linewidth=1)
            ax.scatter(before,i,facecolor='white',edgecolor='#ad572c',s=47,
                       label='Reuse DT' if i==0 else None,zorder=2)
            ax.scatter(after,i,color='#11777a',marker='x',s=39,
                       label='PR projection' if i==0 else None,zorder=3)
        ax.axvline(0,color='#263f4c',linewidth=.8)
        ax.set_yticks(range(len(cases)), ['×'.join(map(str,r['shape'])) for r in cases],fontsize=8)
        ax.set_title({'quant':'DynamicMXQuant','matmul':'Matmul','fa':'Flash Attention'}[op])
        ax.grid(axis='x',alpha=.22)
        ax.spines[['top','right','left']].set_visible(False)
        ax.set_xlabel('Signed error vs PR CA (%)')
    handles,labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles,labels,loc='best',frameon=True,fontsize=8)
    fig.savefig(path,dpi=160)
    plt.close(fig)
