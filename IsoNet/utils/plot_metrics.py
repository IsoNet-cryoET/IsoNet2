
def plot_metrics(metrics, filename):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.set_loglevel("warning") 

    import matplotlib

    from matplotlib.ticker import MaxNLocator
    
    matplotlib.use('agg')

    fig, ax = plt.subplots()
    #with plt.style.context('Solarize_Light2'):
    keys = []
    for k,v in metrics.items():
        if len(v)>0 and k != 'average_loss':
            x = np.arange(len(v))+1
            plt.plot(x, np.array(v), linewidth=2)
            keys.append(k)
    plt.legend(title='metrics', labels=keys)
    #plt.legend(title='metrics', title_fontsize = 13, labels=metrics.keys())
    #if len(tl) > 20:
    #    ma = np.percentile(tl,95)
    #    plt.ylim(top=ma)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("epochs")
    plt.savefig(filename)
    plt.close()