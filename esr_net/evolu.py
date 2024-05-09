import os
import math
import time
import mmcv
import argparse
import numpy as np
import torch.utils.data
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import save_checkpoint
from utils.meter import AverageMeter
from utils.logger import get_root_logger
from utils.seed import set_random_seed
from torch.autograd import Variable
from models.esa_net import ESA_net
from datasets.crowdDataset import CrowdDataset



parser = argparse.ArgumentParser(description='PyTorch CrowdCounting')
parser.add_argument('-train_path', default='/home/houyi/Code/activevcc/gt_data/ADJ3_ZOOM0.35_DR8_AUG18',)
# parser.add_argument('-test_path', default='/home/houyi/datasets/ShanghaiTech/part_B/test_data/ADJ3_ZOOM0.3_DR8')
parser.add_argument('-test_path', default='/home/houyi/Code/activevcc/gt_data/ADJ3_ZOOM0.35_DR8')
parser.add_argument('-lr',default=1e-7,type=float)
parser.add_argument('-scale_times',default=640.0,type=float)
# parser.add_argument('-resume_path',default='/home/houyi/Code/activevcc/weights/esa_net.pth.tar',type=str)
parser.add_argument('-resume_path',default='/home/houyi/Code/activevcc/work_dirs/active_vcc/best_20220319_181727.pth.tar',type=str)
parser.add_argument('-batch_size_train',default=50,type=int)
parser.add_argument('-batch_size_test',default=10,type=int)
parser.add_argument('-start_epoch',default=0,type=int)
parser.add_argument('-epochs',default=1,type=int)
parser.add_argument('-workers',default=10,type=int)
parser.add_argument('-task_id',default='active_vcc',type=str)
args = parser.parse_args()




def train(train_loader, model, criterion, optimizer, epoch):
    logger.info('Epoch: [{0}]\t'.format(epoch))
    losses = AverageMeter()
    model.train()
    for img, density_map, _ in tqdm(train_loader):
        img = Variable(img).cuda()
        density_map = Variable(density_map).cuda()*args.scale_times

        estDensityMap = model(img)
        loss1 = criterion(estDensityMap, density_map)
        loss=loss1
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info('Loss {losses_density.val:.8f} ({losses_density.avg:.8f})\t'.format(losses_density=losses))


def validate(val_loader, model):
    logger.info('begin test:')
    model.eval()
    abs_error=[]
    for i,(img, density_map, _) in enumerate(val_loader):
        with torch.no_grad():
            img = Variable(img).cuda()
            density_map = Variable(density_map).cuda()
            estDensityMap= model(img)
            estDensityMap=estDensityMap*(1.0/args.scale_times)
            diffs1=estDensityMap.sum(dim=[1,2,3])-density_map.data.sum(dim=[1,2,3])
            diffs1=abs(diffs1.cpu().numpy())
            abs_error=np.append(abs_error,diffs1)
    mae=np.mean(abs_error)
    mse=np.std(abs_error)

    logger.info(' * MAE {mae:.3f} '
                ' * MSE {mse:.3f} '.format(mae=mae,mse=mse))
    return mae


def adjust_learning_rate(optimizer, epoch):
    epochs_drop = 5
    args.lr = args.lr * math.pow(0.5, math.floor(epoch / epochs_drop))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr



if __name__ == '__main__':
    # 环境准备
    args.work_dir = os.path.join('./work_dirs',args.task_id)
    mmcv.mkdir_or_exist(os.path.abspath(args.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    set_random_seed(1000)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    ##数据集准备
    train_dataset=CrowdDataset(args.train_path)
    val_dataset=CrowdDataset(args.test_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, num_workers=args.workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_test,num_workers=args.workers)

    add_dataset=CrowdDataset('/home/houyi/datasets/ShanghaiTech/part_B/train_data/ADJ3_ZOOM0.3_DR8_AUG18')
    add_loader = DataLoader(add_dataset, batch_size=args.batch_size_train, num_workers=args.workers, shuffle=True, drop_last=True)

    model = ESA_net()
    model = model.cuda()
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    #主要训练流程准备
    best_prec1 = 1e6
    if args.resume_path:
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume_path, checkpoint['epoch']))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        #
        #
        # train(train_loader, model, criterion, optimizer, epoch)#在这里训练一个批次
        # train(add_loader, model, criterion, optimizer, epoch)#在这里训练一个批次

        prec1 = validate(val_loader, model)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        logger.info(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best=is_best,saving_dir=args.work_dir,saving_prefix=timestamp)