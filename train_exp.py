import os
import math 
import tempfile
import argparse
import random 
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler  
from torchvision import transforms , datasets 
import torch.distributed as dist
import sys 
from tqdm import tqdm
from .TNet import TNet_large, TNet_base, TNet_small
    

 
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=5, help='the number of classes')
parser.add_argument('--epochs', type=int, default=90, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=100, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lrf', type=float, default=0.0001)
parser.add_argument('--seed', default=False, action='store_true', help='fix the initialization of parameters')
parser.add_argument('--aux_train', default=False,  action='store_true')
parser.add_argument('--tensorboard', default=False, action='store_true', help=' use tensorboard for visualization') 
parser.add_argument('--syncBN', type=bool, default=True, help='use syncBN during distrubute learning') 
parser.add_argument('--data_path', type=str,  default="/raid/haowen_yu/dataset/flower") 
parser.add_argument('--model', type=str, default="alexnet")
parser.add_argument('--weights', type=str, default='', help='initial weights path') 
parser.add_argument('--freeze_layers', type=bool, default=False) 
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)') 
parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes') 
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
opt = parser.parse_args()

if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed) # Python random module.	
        os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed) # Numpy module.
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        # 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置:
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # 实际上这个设置对精度影响不大，仅仅是小数点后几位的差别。所以如果不是对精度要求极高，其实不太建议修改，因为会使计算效率降低。
        print('random seed has been fixed')
    seed_torch() 


def init_distrubuted_mode(opt):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ['RANK'])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        opt.rank = int(os.environ['SLURM_PROCID'])
        opt.gpu = opt.rank % torch.cuda.device_count()
    else:
        print('device cannot setup distributed mode')
        opt.distributed = False
        return
    
    opt.distrubuted = True

    torch.cuda.set_device(opt.gpu)
    opt.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(opt.rank, opt.dist_url), flush=True)

    dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
    dist.barrier() #设置一个障碍，即所有rank都运行到这里时才会进行下一步。


def clean_up():
    dist.destroy_process_group()


def is_dist_availble_or_initial():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_availble_or_initial():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_availble_or_initial():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

# 对多个设备上的loss求平均不是为了backward，仅仅是查看做个记录
def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value   
def warmup(optimizer, warm_up_iters, warm_up_factor):
    def f(x):
        """根据step数返回一个学习率倍率因子, x代表step"""
        if x >= warm_up_iters:
            return 1
        
        alpha = float(x) / warm_up_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warm_up_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp=False, lr_method=None):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    train_loss = torch.zeros(1).to(device)
    acc_num = torch.zeros(1).to(device)

    optimizer.zero_grad()
    
    lr_scheduler = None
    if epoch == 0  and lr_method == warmup : 
        warmup_factor = 1.0/1000
        warmup_iters = min(1000, len(data_loader) -1)

        lr_scheduler = warmup(optimizer, warmup_iters, warmup_factor)
    
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    
    # 创建一个梯度缩放标量，以最大程度避免使用fp16进行运算时的梯度下溢 
    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred = model(images.to(device) )
            loss = loss_function(pred, labels.to(device)) 
            pred_class = torch.max(pred, dim=1)[1]  
            acc_num += torch.eq(pred_class, labels.to(device)).sum()
 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += reduce_value(loss, average=True).detach()

        # 在进程中打印平均loss
        if is_main_process():
            info = '[epoch{}]: learning_rate:{:.5f}'.format(
                epoch + 1, 
                optimizer.param_groups[0]["lr"]
            )
            data_loader.desc = info # tqdm 成员 desc
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
 
    return train_loss.item() / (step + 1), acc_num.item() / sample_num
 
        
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证集样本个数
    num_samples = len(data_loader.dataset) 
    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
 
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device) )
        pred_class = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred_class, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
    
    sum_num = reduce_value(sum_num, average=False)
    val_acc = sum_num.item() / num_samples

    return val_acc
 


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distrubuted_mode(opt=args) 
    device = torch.device(args.device)   
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
 
 
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    print(args.data_path)
    train_dataset = datasets.ImageFolder(os.path.join(args.data_path , 'train'), transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path , 'val'), transform=data_transform["val"]) 
 
    # if args.num_classes != train_dataset.num_class:
    #     raise ValueError("dataset have {} classes, but input {}".format(train_dataset.num_class, args.num_classes))
 

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    
    if args.rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
        # save parameters path
    # save_path = os.path.join(os.getcwd(), 'results/weights', args.model)
    # if os.path.exists(save_path) is False:
    #     os.makedirs(save_path)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw )
    # 实例化模型
    model = TNet_large(aux = args.aux_train, num_classes= args.num_classes)

    # 如果存在预训练权重则载入
    if os.path.exists(args.weights):
        weights_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if args.rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    optimizer = optim.Adam(pg, lr=args.lr)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
 
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
 
        mean_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch, lr_method=warmup)
        scheduler.step()

        # validate
        val_acc = evaluate(model=model, data_loader=val_loader, device=device)
        if args.rank == 0: 
            print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, mean_loss, train_acc, val_acc))   
            with open("tnet.txt", 'a') as f: 
                f.writelines(str([round(mean_loss, 3), round(train_acc, 3), round(val_acc, 3)]) + '\n')
            torch.save(model.state_dict(),  "tnet.pth") 
    # 删除临时缓存文件
    if args.rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    clean_up()


if __name__ == '__main__':
 
    main(opt)