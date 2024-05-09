import torch
import shutil
import os

def save_checkpoint(state, is_best, saving_dir, saving_prefix):
    file_path=os.path.join(saving_dir,saving_prefix+'.pth.tar')
    file_path_best=os.path.join(saving_dir,'best_'+saving_prefix+'.pth.tar')
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, file_path_best)