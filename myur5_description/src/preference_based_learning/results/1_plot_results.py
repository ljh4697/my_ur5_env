import numpy as np
import matplotlib.pyplot as plt



DPB_result = np.load('iter300-DPB-seed1.npy')
BA_result = np.load('iter300-batch_active_PBL-seed1.npy')

DPB_cosine_evaluation = DPB_result['eval_cosine']
DPB_simple_regret_evaluation = DPB_result['eval_simple_regret']
DPB_opt_simple_reward = DPB_result['opt_simple_reward']
DPB_cumulative_regret_evaluation = DPB_result['eval_cumulative_regret']

BA_cosine_evaluation = BA_result['eval_cosine']
BA_simple_regret_evaluation = BA_result['eval_simple_regret']
BA_cumulative_regret_evaluation = BA_result['eval_cumulative_regret']


fg = plt.figure(figsize=(10,10))
b = 10
cosine_metric = fg.add_subplot(221)
simple_regret_metric = fg.add_subplot(222)
cumulative_regret_metric = fg.add_subplot(223)



cosine_metric.plot(b*np.arange(len(DPB_cosine_evaluation)), DPB_cosine_evaluation, color='orange', label='DPB', alpha=0.8)
cosine_metric.plot(b*np.arange(len(BA_cosine_evaluation)), BA_cosine_evaluation, color='red', label='BA', alpha=0.8)

cosine_metric.set_ylabel('m')
cosine_metric.set_xlabel('N')
cosine_metric.set_title('cosine metric')
cosine_metric.legend()


simple_regret_metric.plot(b*np.arange(len(DPB_simple_regret_evaluation)), DPB_simple_regret_evaluation, color='orange', label='DPB', alpha=0.8)
simple_regret_metric.plot(b*np.arange(len(DPB_opt_simple_reward)), DPB_opt_simple_reward, color='blue', linestyle='dashed',label='true', alpha=0.8)
simple_regret_metric.plot(b*np.arange(len(BA_simple_regret_evaluation)), BA_simple_regret_evaluation, color='red', label='BA', alpha=0.8)

simple_regret_metric.set_ylabel('m')
simple_regret_metric.set_xlabel('N')
simple_regret_metric.set_title('simple regret')
simple_regret_metric.legend()

cumulative_regret_metric.plot(b*np.arange(len(DPB_cumulative_regret_evaluation)), DPB_cumulative_regret_evaluation, color='orange', label='DPB', alpha=0.8)
cumulative_regret_metric.plot(b*np.arange(len(BA_cumulative_regret_evaluation)), BA_cumulative_regret_evaluation, color='red', label='BA', alpha=0.8)

cumulative_regret_metric.set_ylabel('m')
cumulative_regret_metric.set_xlabel('N')
cumulative_regret_metric.set_title('cumulative regret')
cumulative_regret_metric.legend()

plt.show()

