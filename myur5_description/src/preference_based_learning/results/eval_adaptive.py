import numpy as np
import matplotlib.pyplot as plt



task = 'avoid'

DPB2_DPP = []
DPB_DPP = []


for i in range(1, 11):
    

    DPB_result = np.load('{}/DPP/dpp_score_{:d}.npz'.format(task, i))
    DPB2_DPP.append(DPB_result['DPB2_dpp'])
    DPB_DPP.append(DPB_result['DPB_dpp'])



bar_colors = ['orange', 'purple']
algorithms = ['DPB-adaptive', 'DPB-greedy']
x_pos = np.arange(len(algorithms))


URM = [np.mean(DPB2_DPP), np.mean(DPB_DPP)]
error = [0.1*np.std(DPB2_DPP), 0.1*np.std(DPB_DPP)]
fig, ax = plt.subplots()

ax.bar(x_pos, URM, yerr=error, align='center', color=bar_colors, hatch=['/', '.'], capsize=8)
ax.set_xticks(x_pos)
ax.set_xticklabels(algorithms, fontsize=17)
plt.legend()
plt.yticks(fontsize=15)
plt.savefig('./imgs/{}_determinant_score.png'.format(task))
plt.show()