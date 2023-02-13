import numpy as np
import matplotlib.pyplot as plt

def get_bench_results(task, algo_type:str, num_of_data:int, opt_simple_reward):
    cosine = []
    simple_regret = []
    cumulative_regret = []
    
    for i in range(1, 1+num_of_data):
        bench_result = np.load(task + '/batch_active_PBL/' + f'{task}-iter400-batch_active_PBL-method_{algo_type}-seed{i}.npy')

        cosine.append(bench_result['eval_cosine'])
        simple_regret.append(opt_simple_reward-bench_result['eval_simple_regret'])
        cumulative_regret.append(bench_result['eval_cumulative_regret'])
        

    
    cosine_evaluation = np.mean(cosine,axis=0)
    cosine_evaluation_std = np.std(cosine,axis=0)*0.5
    
    simple_regret_evaluation = np.mean(simple_regret,axis=0)
    simple_regret_evaluation_std = np.std(simple_regret,axis=0)*0.5
    
    cumulative_regret_evaluation = np.mean(cumulative_regret,axis=0)
    cumulative_regret_evaluation_std = np.std(cumulative_regret,axis=0)
    
    
    return (cosine_evaluation, cosine_evaluation_std, 
            simple_regret_evaluation, simple_regret_evaluation_std,
            cumulative_regret_evaluation, cumulative_regret_evaluation_std
            )
    
    
    
    
def plot_cosine_metric(DPB_cosine_evaluation, DPB_cosine_evaluation_std,
                       DPB2_cosine_evaluation, DPB2_cosine_evaluation_std,
                       BA_greedy_cosine_evaluation, BA_greedy_cosine_evaluation_std,
                       BA_medoids_cosine_evaluation, BA_medoids_cosine_evaluation_std,
                       BA_dpp_cosine_evaluation, BA_dpp_cosine_evaluation_std,
                       random_cosine_evaluation, random_cosine_evaluation_std, b=10, task='driver'):
    
    
    
    plt.plot(b*np.arange(len(DPB_cosine_evaluation)), DPB_cosine_evaluation, color='indigo', label='DPB', alpha=1, markevery=0.1, marker='o', markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(DPB2_cosine_evaluation)), DPB2_cosine_evaluation, color='darkorange', label='DPB (adaptive)', alpha=1, markevery=0.1, marker='p', markersize=8.5, linewidth=2)
    
    plt.plot(b*np.arange(len(BA_greedy_cosine_evaluation)), BA_greedy_cosine_evaluation, color='red', label='greedy', alpha=0.4, markevery=0.1, marker='>', markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(BA_medoids_cosine_evaluation)), BA_medoids_cosine_evaluation, color='red', label='medoids', alpha=0.7, markevery=0.1, marker='s', markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(BA_dpp_cosine_evaluation)), BA_dpp_cosine_evaluation, color='red', label='dpp', alpha=1, markevery=0.1, marker='x', markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(random_cosine_evaluation)), random_cosine_evaluation, color='green', label='random', alpha=1, markevery=0.1, marker='d', markersize=8.5, linewidth=2)
    plt.fill_between(b*np.arange(len(DPB_cosine_evaluation)),
                            DPB_cosine_evaluation-DPB_cosine_evaluation_std,
                            DPB_cosine_evaluation+DPB_cosine_evaluation_std,
                            alpha=0.1, color='indigo')
    
    plt.fill_between(b*np.arange(len(DPB2_cosine_evaluation)),
                            DPB2_cosine_evaluation-DPB2_cosine_evaluation_std,
                            DPB2_cosine_evaluation+DPB2_cosine_evaluation_std,
                            alpha=0.1, color='darkorange')
    

    plt.fill_between(b*np.arange(len(BA_greedy_cosine_evaluation)),
                            BA_greedy_cosine_evaluation-BA_greedy_cosine_evaluation_std,
                            BA_greedy_cosine_evaluation+BA_greedy_cosine_evaluation_std,
                            alpha=0.05, color='red')

    plt.fill_between(b*np.arange(len(BA_medoids_cosine_evaluation)),
                            BA_medoids_cosine_evaluation-BA_medoids_cosine_evaluation_std,
                            BA_medoids_cosine_evaluation+BA_medoids_cosine_evaluation_std,
                            alpha=0.08, color='red')

    plt.fill_between(b*np.arange(len(BA_dpp_cosine_evaluation)),
                            BA_dpp_cosine_evaluation-BA_dpp_cosine_evaluation_std,
                            BA_dpp_cosine_evaluation+BA_dpp_cosine_evaluation_std,
                            alpha=0.11, color='red')

    plt.fill_between(b*np.arange(len(random_cosine_evaluation)),
                            random_cosine_evaluation-random_cosine_evaluation_std,
                            random_cosine_evaluation+random_cosine_evaluation_std,
                            alpha=0.1, color='green')

    plt.axvline(x=100, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=200, color='gray', linestyle='--', alpha=0.7)
    plt.ylabel('${m}_{cosine}$', fontsize=20)
    plt.xlabel('Number of iterations ($t$)', fontsize=20)
    plt.ylim((-1, 1))
    plt.xticks(fontsize=15)
    plt.yticks(np.concatenate((np.arange(-1, 0, 0.5),np.arange(0, 1.2, 0.2))),fontsize=15)
    
    #plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig('./saved_graph/{}/{}_cosine_metric.png'.format(task, task))
    plt.show()
    
    
    return

def plot_simple_regret(opt_simple_reward, opt_simple_reward_std,
                       DPB_simple_regret_evaluation, DPB_simple_regret_evaluation_std,
                       BA_greedy_simple_regret_evaluation, BA_greedy_simple_regret_evaluation_std,
                       BA_medoids_simple_regret_evaluation, BA_medoids_simple_regret_evaluation_std,
                       BA_dpp_simple_regret_evaluation, BA_dpp_simple_regret_evaluation_std,
                       random_simple_regret_evaluation, random_simple_regret_evaluation_std, b=10, task='driver'
                       ):
    plt.plot(b*np.arange(len(DPB_simple_regret_evaluation)), DPB_simple_regret_evaluation, color='indigo', label='DPB', alpha=1, marker='o', markevery=0.1, markersize=8.5, linewidth=2)
    #plt.plot(b*np.arange(len(opt_simple_reward)), opt_simple_reward, color='blue', linestyle='dashed',label='true', alpha=0.8, linewidth=2)
    plt.plot(b*np.arange(len(BA_greedy_simple_regret_evaluation)), BA_greedy_simple_regret_evaluation, color='red', label='greedy', alpha=0.4, marker='>', markevery=0.1, markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(BA_medoids_simple_regret_evaluation)), BA_medoids_simple_regret_evaluation, color='red', label='medoids', alpha=0.7, marker='s', markevery=0.1, markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(BA_dpp_simple_regret_evaluation)), BA_dpp_simple_regret_evaluation, color='red', label='dpp', alpha=1, marker='x', markevery=0.1, markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(random_simple_regret_evaluation)), random_simple_regret_evaluation, color='green', label='random', alpha=1, marker='d', markevery=0.1, markersize=8.5, linewidth=2)

    plt.fill_between(b*np.arange(len(DPB_simple_regret_evaluation)),
                            DPB_simple_regret_evaluation-DPB_simple_regret_evaluation_std,
                            DPB_simple_regret_evaluation+DPB_simple_regret_evaluation_std,
                            alpha=0.1, color='indigo')

    plt.fill_between(b*np.arange(len(BA_greedy_simple_regret_evaluation)),
                            BA_greedy_simple_regret_evaluation-BA_greedy_simple_regret_evaluation_std,
                            BA_greedy_simple_regret_evaluation+BA_greedy_simple_regret_evaluation_std,
                            alpha=0.05, color='red')

    plt.fill_between(b*np.arange(len(BA_medoids_simple_regret_evaluation)),
                            BA_medoids_simple_regret_evaluation-BA_medoids_simple_regret_evaluation_std,
                            BA_medoids_simple_regret_evaluation+BA_medoids_simple_regret_evaluation_std,
                            alpha=0.08, color='red')

    plt.fill_between(b*np.arange(len(BA_dpp_simple_regret_evaluation)),
                            BA_dpp_simple_regret_evaluation-BA_dpp_simple_regret_evaluation_std,
                            BA_dpp_simple_regret_evaluation+BA_dpp_simple_regret_evaluation_std,
                            alpha=0.11, color='red')

    plt.fill_between(b*np.arange(len(random_simple_regret_evaluation)),
                            random_simple_regret_evaluation-random_simple_regret_evaluation_std,
                            random_simple_regret_evaluation+random_simple_regret_evaluation_std,
                            alpha=0.1, color='green')

    plt.axvline(x=100, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=200, color='gray', linestyle='--', alpha=0.7)
    plt.ylabel('${m}_{simple}$', fontsize=20)
    plt.xlabel('number of iterations ($t$)', fontsize=20)
    #plt.ylim((np.max(opt_simple_reward)+0.5, np.min(random_simple_regret_evaluation)-0.5))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    #plt.legend(fontsize=13, framealpha=0)
    plt.tight_layout()
    plt.savefig('./saved_graph/{}/{}_simple_regret.png'.format(task, task))
    
    plt.show()
    
    return


def plot_cumulative_regret(DPB_cumulative_regret_evaluation, DPB_cumulative_regret_evaluation_std,
                           BA_greedy_cumulative_regret_evaluation, BA_greedy_cumulative_regret_evaluation_std,
                           BA_medoids_cumulative_regret_evaluation, BA_medoids_cumulative_regret_evaluation_std,
                           BA_dpp_cumulative_regret_evaluation, BA_dpp_cumulative_regret_evaluation_std,
                           random_cumulative_regret_evaluation, random_cumulative_regret_evaluation_std, b=10, task='driver'):
        
    plt.plot(b*np.arange(len(DPB_cumulative_regret_evaluation)), DPB_cumulative_regret_evaluation, color='indigo', label='DPB', alpha=1, markevery=0.1, marker='o', markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(BA_greedy_cumulative_regret_evaluation)), BA_greedy_cumulative_regret_evaluation, color='red', label='greedy', alpha=0.4, markevery=0.1, marker='>', markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(BA_medoids_cumulative_regret_evaluation)), BA_medoids_cumulative_regret_evaluation, color='red', label='medoids', alpha=0.6, markevery=0.1, marker='s', markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(BA_dpp_cumulative_regret_evaluation)), BA_dpp_cumulative_regret_evaluation, color='red', label='dpp', alpha=1, markevery=0.1, marker='x', markersize=8.5, linewidth=2)
    plt.plot(b*np.arange(len(random_cumulative_regret_evaluation)), random_cumulative_regret_evaluation, color='green', label='random', alpha=1, markevery=0.1, marker='d', markersize=8.5, linewidth=2)


    plt.fill_between(b*np.arange(len(DPB_cumulative_regret_evaluation)),
                            DPB_cumulative_regret_evaluation-DPB_cumulative_regret_evaluation_std,
                            DPB_cumulative_regret_evaluation+DPB_cumulative_regret_evaluation_std,
                            alpha=0.1, color='indigo')

    plt.fill_between(b*np.arange(len(BA_greedy_cumulative_regret_evaluation)),
                            BA_greedy_cumulative_regret_evaluation-BA_greedy_cumulative_regret_evaluation_std,
                            BA_greedy_cumulative_regret_evaluation+BA_greedy_cumulative_regret_evaluation_std,
                            alpha=0.05, color='red')

    plt.fill_between(b*np.arange(len(BA_medoids_cumulative_regret_evaluation)),
                            BA_medoids_cumulative_regret_evaluation-BA_medoids_cumulative_regret_evaluation_std,
                            BA_medoids_cumulative_regret_evaluation+BA_medoids_cumulative_regret_evaluation_std,
                            alpha=0.08, color='red')

    plt.fill_between(b*np.arange(len(BA_dpp_cumulative_regret_evaluation)),
                            BA_dpp_cumulative_regret_evaluation-BA_dpp_cumulative_regret_evaluation_std,
                            BA_dpp_cumulative_regret_evaluation+BA_dpp_cumulative_regret_evaluation_std,
                            alpha=0.11, color='red')

    plt.fill_between(b*np.arange(len(random_cumulative_regret_evaluation)),
                            random_cumulative_regret_evaluation-random_cumulative_regret_evaluation_std,
                            random_cumulative_regret_evaluation+random_cumulative_regret_evaluation_std,
                            alpha=0.1, color='green')



    plt.axvline(x=100, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=200, color='gray', linestyle='--', alpha=0.7)
    plt.ylabel('${m}_{cumulative}$', fontsize=20)
    plt.xlabel('number of iterations ($t$)', fontsize=20)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15, framealpha=0)
    plt.tight_layout()
    plt.savefig('./saved_graph/{}/{}_cumulative_regret.png'.format(task, task))
    
    plt.show()

