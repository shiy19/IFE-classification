import os.path as osp
import os
import argparse
import numpy as np
from tqdm import tqdm
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import (get_dataset_spectral_mtl, pprint, set_gpu, mkdir, compute_fnr, dl2label)
from sklearn import metrics


def get_args():    
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--backbone', type=str, default='resnet18', 
                        choices=['resnet18'])    
    # optimization parameters
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--scheduler', type=str, default='cos', help='learning rate scheduler (default: cos)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')      
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs (default: 5)')   
    parser.add_argument('--decay', type=float, default=5e-4)    

    # regularizer parameters
    parser.add_argument('--lmda1', type=float, default=0.3)
    parser.add_argument('--lmda2', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=5)
    parser.add_argument('--share_layer_order', type=int, default=3, choices=[-1,0,1,2,3,4], help="Share layers until share layer order.")
    
    # other choices
    parser.add_argument('--no_train', action="store_true")
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    
    set_gpu(args.gpu)
  
    return args

def set_seed(a):
    random.seed(a)
    os.environ['PYTHONHASHSEED'] = str(a)
    np.random.seed(a)   
    torch.manual_seed(a)
    torch.cuda.manual_seed(a)
    torch.cuda.manual_seed_all(a)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_savepath():
    global args
    save_path1 = '-'.join([args.backbone, 'share{}'.format(args.share_layer_order)])

    save_path1 = osp.join('../result',save_path1)

    save_path2 = 'Bsz{}-Epoch-{}-Cos-lr{}decay{}'.format(args.batch_size, args.max_epoch, args.lr, args.decay)

    save_path2 += 'lmda1{}lmda2{}tau{}'.format(args.lmda1,args.lmda2,args.tau)

    if args.warmup_epochs > 0:
        save_path2 += 'Warmup{}'.format(args.warmup_epochs)

    args.save_path = osp.join(save_path1, save_path2)
    mkdir(args.save_path)  


def save_model(model, name):
    global args
    torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    

def get_optimizer(args, model, train_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay, nesterov=True)
    
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (args.max_epoch - args.warmup_epochs) * len(train_loader))
    
    if args.warmup_epochs > 0:
        lr_schedule_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 * epoch / (args.warmup_epochs * len(train_loader)))    #???
    else:
        lr_schedule_warmup = None
    
    return optimizer, lr_schedule, lr_schedule_warmup

def reg_inside(attn, label, tau):
    reg = 0
    for i in [0,1]:
        dist_s = F.softmax(attn[-i-1]/tau,dim=2).detach()
        dist_t = F.softmax(attn[-i-1]*tau,dim=2).detach()
        dist_true = F.softmax(attn[-i-1],dim=2)
        kl_loss = nn.KLDivLoss(reduction="sum")
        reg += 0.5*kl_loss(dist_true[label==0].log(),dist_s[label==0]) # This is for imbalance
        reg += kl_loss(dist_true[label!=0].log(),dist_t[label!=0])
    return 0.01*reg

def reg_between(attn1, attn2, label):
    reg = 0
    flag = (label!=0)&(label!=1)&(label!=2)
    for i in [0,1]:
        kl_loss = nn.KLDivLoss(reduction="sum")
        dist1 = F.softmax(attn1[-i-1][flag],dim=2)
        dist2 = F.softmax(attn2[-i-1][flag],dim=2)
        tmp = (dist1 + dist2)/2
        reg += 0.5*(kl_loss(tmp.log(),dist1)+kl_loss(tmp.log(),dist2))
    return 100*reg


def train_model(args, model, train_loader, criterion, optimizer, lr_scheduler, epoch):    
    model.train()
    global global_count, trlog
    for i, batch in enumerate(train_loader):
        data = batch[0].cuda()
        batch_label1 = batch[1].cuda()
        batch_label2 = batch[2].cuda()
        batch_label = batch[3].cuda()
        logit1, logit2 = model(data)
        attn1, attn2 = model.get_attention()

        loss1 = criterion(logit1, batch_label1)
        loss2 = criterion(logit2, batch_label2)
        loss = loss1 + loss2
        if args.lmda1 != 0:
            reg_in1 = reg_inside(attn1,batch_label1,args.tau)
            reg_in2 = reg_inside(attn2,batch_label2,args.tau)
            reg_in = reg_in1 + reg_in2
            loss += args.lmda1* reg_in
        if args.lmda2 != 0:
            reg_be = reg_between(attn1,attn2,batch_label)
            loss += args.lmda2*reg_be

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3, norm_type=2)
        optimizer.step()
    
        lr_scheduler.step()       
        global_count = global_count + 1        
        
        torch.cuda.empty_cache()

        if i % args.log_interval == 0:
            print('Train Epoch: {} \t' 'lr={:.4g}'.format(epoch, optimizer.param_groups[0]['lr']))        
    return model
    
def validate_model(args, model, val_loader, epoch):
    print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    model.eval()
    test_logit1, test_logit2 = [], []
    test_label, test_label1, test_label2 = [], [], []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            data = batch[0].cuda()
            batch_label1 = batch[1].cuda()
            batch_label2 = batch[2].cuda()
            batch_label = batch[3].cuda()
            logit1, logit2 = model(data)

            test_logit1.append(logit1)
            test_logit2.append(logit2)

            test_label1.append(batch_label1)
            test_label2.append(batch_label2)
            test_label.append(batch_label)
            
    test_logit1 = torch.cat(test_logit1, 0)
    test_logit2 = torch.cat(test_logit2, 0)
    test_label1 = torch.cat(test_label1, 0)
    test_label2 = torch.cat(test_label2, 0)
    test_label = torch.cat(test_label, 0)

    test_pred1 = torch.argmax(test_logit1, dim=1)
    test_pred2 = torch.argmax(test_logit2, dim=1)
    test_pred = dl2label(test_pred1, test_pred2)
    acc1 = (test_pred1 == test_label1).type(torch.cuda.FloatTensor).mean().item()
    acc2 = (test_pred2 == test_label2).type(torch.cuda.FloatTensor).mean().item()
    acc = (test_pred == test_label).type(torch.cuda.FloatTensor).mean().item()      
    print('epoch {} acc1={:.4f}, acc2={:.4f}, acc={:.4f}'.format(epoch, acc1, acc2, acc))
    if acc >= trlog['max_acc']:
        trlog['max_acc'] = acc
        trlog['max_acc_epoch'] = epoch
        save_model(model, 'max_acc')                  


def test_model(args, model, val_loader):
    model.eval()
    test_logit1, test_logit2 = [], []
    test_label, test_label1, test_label2 = [], [], []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            
            data = batch[0].cuda()
            batch_label1 = batch[1].cuda()
            batch_label2 = batch[2].cuda()
            batch_label = batch[3].cuda()
            logit1, logit2 = model(data)

            test_logit1.append(logit1)
            test_logit2.append(logit2)

            test_label1.append(batch_label1)
            test_label2.append(batch_label2)
            test_label.append(batch_label)
            
    test_logit1 = torch.cat(test_logit1, 0)
    test_logit2 = torch.cat(test_logit2, 0)
    test_label1 = torch.cat(test_label1, 0)
    test_label2 = torch.cat(test_label2, 0)
    test_label = torch.cat(test_label, 0)

    test_pred1 = torch.argmax(test_logit1, dim=1)
    test_pred2 = torch.argmax(test_logit2, dim=1)
    test_pred = dl2label(test_pred1, test_pred2)
    acc1 = (test_pred1 == test_label1).type(torch.cuda.FloatTensor).mean().item()
    acc2 = (test_pred2 == test_label2).type(torch.cuda.FloatTensor).mean().item()
    acc = (test_pred == test_label).type(torch.cuda.FloatTensor).mean().item()  
    
    from sklearn.metrics import confusion_matrix
    confusion_arr1 = confusion_matrix(test_label1.cpu().numpy(),test_pred1.cpu().numpy())
    confusion_arr2 = confusion_matrix(test_label2.cpu().numpy(),test_pred2.cpu().numpy())
    confusion_arr = confusion_matrix(test_label.cpu().numpy(),test_pred.cpu().numpy())   
    return test_pred, test_label, acc, acc1, acc2, confusion_arr, confusion_arr1, confusion_arr2


if __name__ == '__main__':
    args = get_args()
    set_seed(42)
    torch.cuda.empty_cache()
    set_savepath()
    pprint(vars(args))   
            
    ## Prepare Stage
    trainset, valset, train_loader, val_loader = get_dataset_spectral_mtl(args)

    from model.models.Vanilla import Classifier_Spectal_MTL as Classifier

    model = Classifier(args) 
              
    criterion = nn.CrossEntropyLoss().cuda()
        
    optimizer, lr_schedule, lr_schedule_warmup = get_optimizer(args, model, train_loader)
    
    model = model.cuda()
        
    trlog = {}
    trlog['args'] = vars(args)
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0
    
    # # timer = Timer()
    global_count = 0
    
    # Training Stage
    # warmup
    if not args.no_train:
        for warmup_epoch in range(args.warmup_epochs):
            model = train_model(args, model, train_loader, criterion, optimizer, lr_schedule_warmup, warmup_epoch)
        if args.warmup_epochs > 0:
            validate_model(args, model, val_loader, -1) 
        
        for epoch in range(args.warmup_epochs, args.max_epoch):
            tic = time.time()
            model = train_model(args, model, train_loader, criterion, optimizer, lr_schedule, epoch)
            validate_model(args, model, val_loader, epoch)
            elapsed = time.time() - tic
            print(f'Epoch: {epoch}, Time cost: {elapsed}')
        print('best_acc:{}'.format(trlog['max_acc']))  

        #save tr_log
        for key in trlog.keys():
            value = trlog[key]
            with open(os.path.join(args.save_path, 'log.txt'), 'a+') as f:
                f.write('{}:{}\n'.format(key, value))

    # Test Stage
    model.load_state_dict(torch.load(osp.join(args.save_path,'max_acc.pth'))['params'])
    pred, true, acc, acc1, acc2, confusion_arr, confusion_arr1, confusion_arr2 = test_model(args, model, val_loader)
    y_pred = pred.cpu().numpy()
    y_true = true.cpu().numpy()
    fnr = compute_fnr(y_true,y_pred)
    print(fnr)
    with open(osp.join(args.save_path,'result.txt'),'w') as f:
        print("fnr:{:.4f}".format(fnr),file=f)
        print("acc1:{:.4f}".format(acc1), file=f)
        print(confusion_arr1, file=f)
        print("acc2:{:.4f}".format(acc2), file=f)
        print(confusion_arr2, file=f)
        print("acc:{:.4f}".format(acc), file=f)
        print(confusion_arr, file=f)
        print(metrics.classification_report(y_true,y_pred,digits=4),file=f)