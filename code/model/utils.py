import os
import time
import errno
import pprint
import torch
from torch.utils.data import DataLoader

# --- optimization helper --- 

def mkdir(path):
    """make dir exists okay"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
        
def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)
    
#    --- method helper ---
def get_dataset_spectral_mtl(args):
    from model.dataloader.spectral import Spectral_MDC as Dataset
    trainset = Dataset('train', aug=True, large_img=True)
    valset = Dataset('test', aug=False, large_img=True)    
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return trainset, valset, train_loader, val_loader

def compute_fnr(epoch_label, best_epoch_pred):
    fn = sum((epoch_label!=0) & (best_epoch_pred==0))
    all_pos = sum(epoch_label!=0)
    fnr = fn/all_pos
    return fnr

def dl2label(label1,label2):
    res = []
    for i in range(len(label1)):
        if label2[i] == 1:
            if label1[i] == 0:
                label = 1
            elif label1[i] == 1:
                label = 3
            elif label1[i] == 2:
                label = 5
            elif label1[i] == 3:
                label = 7
        elif label2[i] == 2:
            if label1[i] == 0:
                label = 2
            elif label1[i] == 1:
                label = 4
            elif label1[i] == 2:
                label = 6
            elif label1[i] == 3:
                label = 8
        else:
            label = 0
        res.append(label)
    return torch.Tensor(res).cuda()