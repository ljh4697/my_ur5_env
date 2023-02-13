import numpy as np
import matplotlib.pyplot as plt
import os
import sys





best_perform = str()
consine_best_score = 0
second_best_score = 0

task = 'driver'

save_path = f'{task}/DPB'




for f in os.listdir(save_path):
    
    if 'ndriver' in f and 'seed1' in f:
        filepath = save_path + '/' + f
        
        n, s = f.split('seed')
        # if s != '5.npy':
        #     continue
        
        if 'lambda' not in n:
            continue
        
        if len(f.split('-')) >= 2:
            
            if 'v' in f:
            
                if f.split('-')[2] == "DPB":
                    DPB_result = np.load(save_path + '/' + f)
                    if np.sum(DPB_result['eval_simple_regret']) > consine_best_score:
                        consine_best_score = np.sum(DPB_result['eval_simple_regret'])
                        best_perform = f
                    elif consine_best_score > np.sum(DPB_result['eval_simple_regret']) and np.sum(DPB_result['eval_simple_regret']) > second_best_score:
                        second_best_score = np.sum(DPB_result['eval_simple_regret'])
                        secone_best_perform = f
                    
                
print(best_perform)
print(secone_best_perform)
                
                