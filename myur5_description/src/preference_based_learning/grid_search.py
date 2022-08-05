from bench_demos import bench_experiment

###########################
batch_active_params = {
    "samples_num":1000
}

##############################
DPB_params = {
    "exploration_weight":0.02,
    "discounting_factor":0.93,
    "action_U":1.4,
    "param_U":1,
    "regularized_lambda":0.1,
    "reward_U":1,
    "delta":0.6,
    "c_mu":1/5,
    "k_mu":1/4
}

############################



if __name__ == "__main__":
    for a in range(25):
        alpha = 0.02*(a+1)
        bench_experiment("driver", "greedy",
                         N=900, b=10, 
                         batch_active_params=batch_active_params,
                         DPB_params=DPB_params,
                         num_randomseeds=5)