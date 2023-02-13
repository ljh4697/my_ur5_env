import os

from cv2 import split


dir_path = './driver/DPB'

for f in os.listdir(dir_path):
    
    os.remove(dir_path + '/' + f)

