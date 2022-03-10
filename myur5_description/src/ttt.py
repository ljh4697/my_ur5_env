import numpy as np
import os



def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    psi_path = os.path.join(dir_path, 'sampled_trajectories/psi_set.npz')
    psi_set = np.load(psi_path)
    
    
    print(len(psi_set['PHI_A']))
    print(len(psi_set['PSI_SET']))
    
    
    print(psi_set['PHI_A'][0])
    print(psi_set['PHI_B'][0])
    
    print(psi_set['PSI_SET'][1])


# for i in range(200):
#     print(i//2)

if __name__ == "__main__":
    main()