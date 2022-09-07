import numpy as np



import matplotlib.pyplot as plt




    
plt.plot(0, 0, color='indigo', label='DPB', alpha=1)
plt.plot(0, 0, color='red', label='greedy', alpha=0.4)
plt.plot(0, 0, color='red', label='medoids', alpha=0.7)
plt.plot(0, 0, color='red', label='dpp', alpha=1)
plt.plot(0, 0, color='green', label='random', alpha=1)


plt.yticks(np.concatenate((np.arange(-1, 0, 0.5),np.arange(0, 1.2, 0.2))),fontsize=13)

#plt.legend(fontsize=13)
plt.tight_layout()
legend = plt.legend(ncol=5, framealpha=1, frameon=False)
    


def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)
plt.show()