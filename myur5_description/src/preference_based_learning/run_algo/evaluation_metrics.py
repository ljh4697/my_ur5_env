import numpy as np

def cosine_metric(w, true_w):
    
    result = np.dot(w, true_w)/(np.linalg.norm(w)*np.linalg.norm(true_w))
    
    return result


def simple_regret(w, true_w):
    
    return
    
def regret(w, true_w):
    
    return