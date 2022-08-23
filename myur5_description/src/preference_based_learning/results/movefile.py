import os
import shutil
import sys




print(os.path.dirname(__file__))





for f in os.listdir():
    if len(f.split('-'))>=2:
        if f.split('-')[2] == 'DPB':
            print(os.path.dirname(__file__)+f)
            shutil.move(os.path.dirname(__file__)+'/'+f, os.path.dirname(__file__)+'/DPB/'+f)
        elif f.split('-')[2] == 'batch_active_PBL':
            shutil.move(os.path.dirname(__file__)+'/'+f, os.path.dirname(__file__)+'/batch_active_PBL/'+f)
            