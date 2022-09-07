import numpy as np
import matplotlib.pyplot as plt
import os
import sys





best_perform = str()
second_perform = str()
consine_best_score = 0
second_best_score = 0
third = 0
task = 'tosser'

save_path = f'{task}/DPB'



params_set = []



for f in os.listdir(save_path):
    filepath = save_path + '/' + f
    
    
    params, seed = filepath.split('seed')
    
    params_set.append(params)
    
avgparams_set = []
for f in params_set:
    
    if params_set.count(f) == 10:
        
        if avgparams_set.count(f) == 0:
        
            avgparams_set.append(f)
    
    
    

mean_cosine = []


for param in avgparams_set:
    
    
    for i in range(1, 11):
        
        DPB_result = np.load(param + 'seed' + str(i) + ".npy")

        mean_cosine.append(DPB_result['eval_cosine'])
        
        #print(DPB_result['eval_cosine'])
        
    consine_score = np.mean(mean_cosine, axis=0)
    sum_cosine = np.sum(consine_score)
    
    
    if sum_cosine > consine_best_score:
        consine_best_score = sum_cosine
        best_perform = param
        
    elif sum_cosine>second_best_score:
        second_best_score =consine_best_score
        second_perform = param
        
    elif sum_cosine>third:
        third =consine_best_score
        third_perform = param        
        
    mean_cosine = []


print(best_perform)
print(second_perform)
print(third_perform)
        
        
# print(param)
        
    
    
    # if len(f.split('-')) >= 2:
        
    #     if f.split('-')[2] == "DPB":
    #         DPB_result = np.load(save_path + '/' + f)
    #         if np.sum(DPB_result['eval_cosine']) > consine_best_score:
    #             consine_best_score = np.sum(DPB_result['eval_cosine'])
    #             best_perform = f
    #         elif consine_best_score > np.sum(DPB_result['eval_cosine']) and np.sum(DPB_result['eval_cosine']) > second_best_score:
    #             second_best_score = np.sum(DPB_result['eval_cosine'])
    #             secone_best_perform = f
                
                
#print(best_perform)
#print(secone_best_perform)
                
                