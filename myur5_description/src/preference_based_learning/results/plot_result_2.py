import numpy as np
import matplotlib.pyplot as plt
from plot_utils import get_bench_results, plot_cosine_metric, plot_simple_regret, plot_cumulative_regret



task = 'tosser'

delta = 0.70
alpha = 0.006 # 0.0005 for avoid 0.0002 for driver
gamma = 0.95
lamb = 0.1
DPB_cosine = []
DPB_simple_regret = []
DPB_cumulative_regret = []
DPB_opt_simple_reward = []


for i in range(1, 2):
    
    DPB_result = np.load(task + '/DPB/' + '{:}-iter400-DPB-delta{:.2f}-alpha{:.4f}-gamma{:.3f}-lambda{:.3f}-seed{:d}.npy'.format(task, delta, alpha, gamma, lamb, i))

    DPB_cosine.append(DPB_result['eval_cosine'])
    DPB_simple_regret.append(DPB_result['eval_simple_regret'])
    DPB_opt_simple_reward.append(DPB_result['opt_simple_reward'])
    DPB_cumulative_regret.append(DPB_result['eval_cumulative_regret'])
        
DPB_cosine_evaluation = np.mean(DPB_cosine, axis=0)
DPB_cosine_evaluation_std = np.std(DPB_cosine, axis=0)

DPB_simple_regret_evaluation = np.mean(DPB_simple_regret, axis=0)
DPB_simple_regret_evaluation_std = np.std(DPB_simple_regret, axis=0)

opt_simple_reward = np.mean(DPB_opt_simple_reward, axis=0)
opt_simple_reward_std = np.std(DPB_opt_simple_reward, axis=0)

DPB_cumulative_regret_evaluation = np.mean(DPB_cumulative_regret, axis=0)
DPB_cumulative_regret_evaluation_std = np.std(DPB_cumulative_regret, axis=0)




# bench marking algorithms' results
(BA_greedy_cosine_evaluation, BA_greedy_cosine_evaluation_std,
 BA_greedy_simple_regret_evaluation, BA_greedy_simple_regret_evaluation_std,
 BA_greedy_cumulative_regret_evaluation, BA_greedy_cumulative_regret_evaluation_std) = get_bench_results(task, 'greedy', 10)

(BA_medoids_cosine_evaluation, BA_medoids_cosine_evaluation_std,
 BA_medoids_simple_regret_evaluation, BA_medoids_simple_regret_evaluation_std,
 BA_medoids_cumulative_regret_evaluation, BA_medoids_cumulative_regret_evaluation_std) = get_bench_results(task, 'medoids', 10)

(BA_dpp_cosine_evaluation, BA_dpp_cosine_evaluation_std,
 BA_dpp_simple_regret_evaluation, BA_dpp_simple_regret_evaluation_std,
 BA_dpp_cumulative_regret_evaluation, BA_dpp_cumulative_regret_evaluation_std) = get_bench_results(task, 'dpp', 10)

(random_cosine_evaluation, random_cosine_evaluation_std,
 random_simple_regret_evaluation, random_simple_regret_evaluation_std,
 random_cumulative_regret_evaluation, random_cumulative_regret_evaluation_std) = get_bench_results(task, 'random', 10)






# plot_cosine_metric(DPB_cosine_evaluation, DPB_cosine_evaluation_std,
#                    BA_greedy_cosine_evaluation, BA_greedy_cosine_evaluation_std,
#                    BA_medoids_cosine_evaluation, BA_medoids_cosine_evaluation_std,
#                    BA_dpp_cosine_evaluation, BA_dpp_cosine_evaluation_std,
#                    random_cosine_evaluation, random_cosine_evaluation_std)



# plot_simple_regret(opt_simple_reward, opt_simple_reward,
#                    DPB_simple_regret_evaluation, DPB_simple_regret_evaluation_std,
#                    BA_greedy_simple_regret_evaluation, BA_greedy_simple_regret_evaluation_std,
#                    BA_medoids_simple_regret_evaluation, BA_medoids_simple_regret_evaluation_std,
#                    BA_dpp_simple_regret_evaluation, BA_dpp_simple_regret_evaluation_std,
#                    random_simple_regret_evaluation, random_simple_regret_evaluation_std)

# plot_cumulative_regret(DPB_cumulative_regret_evaluation, DPB_cumulative_regret_evaluation_std,
#                    BA_greedy_cumulative_regret_evaluation, BA_greedy_cumulative_regret_evaluation_std,
#                    BA_medoids_cumulative_regret_evaluation, BA_medoids_cumulative_regret_evaluation_std,
#                    BA_dpp_cumulative_regret_evaluation, BA_dpp_cumulative_regret_evaluation_std,
#                    random_cumulative_regret_evaluation, random_cumulative_regret_evaluation_std)













# #### subplot
fg = plt.figure(figsize=(10,10))
b = 10
cosine_metric = fg.add_subplot(221)
simple_regret_metric = fg.add_subplot(222)
cumulative_regret_metric = fg.add_subplot(223)



cosine_metric.plot(b*np.arange(len(DPB_cosine_evaluation)), DPB_cosine_evaluation, color='orange', label='DPB', alpha=0.8)
cosine_metric.plot(b*np.arange(len(BA_greedy_cosine_evaluation)), BA_greedy_cosine_evaluation, color='red', label='greedy', alpha=0.4)
cosine_metric.plot(b*np.arange(len(BA_medoids_cosine_evaluation)), BA_medoids_cosine_evaluation, color='red', label='medoids', alpha=0.6)
cosine_metric.plot(b*np.arange(len(BA_dpp_cosine_evaluation)), BA_dpp_cosine_evaluation, color='red', label='dpp', alpha=0.8)
cosine_metric.plot(b*np.arange(len(random_cosine_evaluation)), random_cosine_evaluation, color='green', label='random', alpha=0.8)

cosine_metric.axvline(x=100, color='gray', linestyle='--', alpha=0.7)
cosine_metric.axvline(x=200, color='gray', linestyle='--', alpha=0.7)
cosine_metric.set_ylabel('m')
cosine_metric.set_xlabel('N')
cosine_metric.set_title('cosine metric')
cosine_metric.set_ylim((-1, 1))
cosine_metric.legend()


simple_regret_metric.plot(b*np.arange(len(DPB_simple_regret_evaluation)), DPB_simple_regret_evaluation, color='orange', label='DPB', alpha=0.8)
simple_regret_metric.plot(b*np.arange(len(opt_simple_reward)), opt_simple_reward, color='blue', linestyle='dashed',label='true', alpha=0.8)
simple_regret_metric.plot(b*np.arange(len(BA_greedy_simple_regret_evaluation)), BA_greedy_simple_regret_evaluation, color='red', label='greedy', alpha=0.4)
simple_regret_metric.plot(b*np.arange(len(BA_medoids_simple_regret_evaluation)), BA_medoids_simple_regret_evaluation, color='red', label='medoids', alpha=0.6)
simple_regret_metric.plot(b*np.arange(len(BA_dpp_simple_regret_evaluation)), BA_dpp_simple_regret_evaluation, color='red', label='dpp', alpha=0.8)
simple_regret_metric.plot(b*np.arange(len(random_simple_regret_evaluation)), random_simple_regret_evaluation, color='green', label='random', alpha=0.8)

simple_regret_metric.axvline(x=100, color='gray', linestyle='--', alpha=0.7)
simple_regret_metric.axvline(x=200, color='gray', linestyle='--', alpha=0.7)
simple_regret_metric.set_ylabel('m')
simple_regret_metric.set_xlabel('N')
simple_regret_metric.set_title('simple regret')
simple_regret_metric.legend()

cumulative_regret_metric.plot(b*np.arange(len(DPB_cumulative_regret_evaluation)), DPB_cumulative_regret_evaluation, color='orange', label='DPB', alpha=0.8)
cumulative_regret_metric.plot(b*np.arange(len(BA_greedy_cumulative_regret_evaluation)), BA_greedy_cumulative_regret_evaluation, color='red', label='greedy', alpha=0.4)
cumulative_regret_metric.plot(b*np.arange(len(BA_medoids_cumulative_regret_evaluation)), BA_medoids_cumulative_regret_evaluation, color='red', label='medoids', alpha=0.6)
cumulative_regret_metric.plot(b*np.arange(len(BA_dpp_cumulative_regret_evaluation)), BA_dpp_cumulative_regret_evaluation, color='red', label='dpp', alpha=0.8)
cumulative_regret_metric.plot(b*np.arange(len(random_cumulative_regret_evaluation)), random_cumulative_regret_evaluation, color='green', label='random', alpha=0.8)


cumulative_regret_metric.axvline(x=100, color='gray', linestyle='--', alpha=0.7)
cumulative_regret_metric.axvline(x=200, color='gray', linestyle='--', alpha=0.7)
cumulative_regret_metric.set_ylabel('m')
cumulative_regret_metric.set_xlabel('N')
cumulative_regret_metric.set_title('cumulative regret')
cumulative_regret_metric.legend()

plt.show()

