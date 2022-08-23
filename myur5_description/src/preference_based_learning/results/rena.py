import os

from cv2 import split




for f in os.listdir('./driver/batch_active_PBL'):
    if f.split('-')[3] == 'methodmedoids':
        continue
    elif f.split('-')[3] == 'method_greedy':
        words = f.split('-')
        words[0] ='driver'
        os.rename('./driver/batch_active_PBL/'+f,'./driver/batch_active_PBL/'+ "-".join(words))
        

    else:
        words = f.split('-')
        words[0] ='driver'
        os.rename('./driver/batch_active_PBL/'+f,'./driver/batch_active_PBL/'+ "-".join(words))
        
#    print(f)


