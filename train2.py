'''
greatest artists always hide themselves in their work
author:刘津铭
ds : 20230527
email: 22210980054@m.fudan.edu.cn
feel free to copy my code ,since I sometimes do the similar thing in terms of deep learning :)
'''
from vit_pytorch import ViT # vision transformer net
from DataProcess import get_data, MyDataset,get_data_cutout,get_data_mixup,get_data_cutmix# 引入本地文件DataProcess.py
# from CNN_dropout import cnn # 引入本地文件 CNN.py

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# 使用gpu加速


class nn_processor:
    def __init__(self, train_loader, valid_loader=None, valid_list=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_list = valid_list

    def train(self, net, lr=0.01, EPOCH=40, max_iter=500, save_iter=500, plot_iter=2000, first_iter=0,print_iter = 100,
              loss_func=nn.BCEWithLogitsLoss(), save_path = False,test_dataset = False,roll = True,weight_decay = 0,lr_beta = 1):
        # train_writer = SummaryWriter('logs_train')
        # test_writer = SummaryWriter('logs_test')
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=weight_decay)  # 这里没有加入encoder.parameters(),所以自编码网络的参数不会改变
        net = net.to(device)
        i = 0
        loss_train_list = list()
        loss_valid_list = list()
        iter_list = list()
        stop = False
        for epoch in tqdm(range(EPOCH)):
            # if epoch % 2 == 0: # 学习率随epoch衰减策略
            #     for p in optimizer.param_groups:
            #         p['lr'] = lr_beta
            if stop == True:
                break
            print(f'\nepoch {epoch}')
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(device), y.to(device)
                y_pre = net(x)
                # output1 = output1.to(torch.float)
                # y = y.to(torch.float)
                # output2 = output2.to(torch.float)
                # y2 =y2.to(torch.float)
                loss = loss_func(y_pre, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1

                if i % print_iter == 0:
                    print(f'\nepoch:{epoch+1}\niteration: {i+first_iter}')
                    print('train loss:', float(loss))
                    for k,(xv,yv) in enumerate(self.valid_loader):
                        xv, yv = xv.to(device), yv.to(device)
                        y_prev = net(xv)
                        lossv = loss_func(y_prev, yv)
                        print('valid loss:', float(lossv))
                        break
                    # train_writer.add_scalar('loss', loss, i)
                    # test_writer.add_scalar('loss', lossv, i)
                    if test_dataset:
                        accuracy = predict(net, test_dataset)
                    if i >= max_iter:  # 达到最大迭代，保存模型
                        stop = True
                        torch.save(net.state_dict(), f'{save_path}/{i+first_iter}.pth')
                        print('model saved!')
                        break
                    if save_path: # 如果指定了保存模型和路径
                        if i % save_iter == 0:  # 临时保存
                            if i != save_iter and i % 50000 != 0 and roll == True:
                                os.remove(f'{save_path}/{i+first_iter-save_iter}.pth')
                            torch.save(net.state_dict(), f'{save_path}/{i+first_iter}.pth')
                            print(f'model temp {i+first_iter} saved!')
                    #
                    # print('val ac:', DataProcess().predict(net, valid_list))

if __name__ == '__main__':
    batch_size = 256
    first_iter = 100000 # 更改
    print_iter = 100
    save_iter = 10000 # 更改
    Epoch = 2000000
    max_iter = 200000 # 更改
    lr = 1e-4
    lr_beta = 1 # lr 衰减策略
    # L2 = 0.005 # L2正则化,用于应对CFAR-100这种逆天过拟合数据集, CNN.py 0.001 -> 0.41
    L2 = 0.003 # 初始 1e-3似乎也还行，改成0.0075试一试
    save_path = 'model_save' # 保存模型的文件夹，如需保存模型请注释掉下一行
    loss_func = nn.CrossEntropyLoss()
    # save_path = False # 不进行任何保存操作

    train_list = get_data()
    # train_list = get_data_cutout(0.65)
    # train_list = get_data_cutmix()
    valid_list = get_data(istrain = False)
    TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
        transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
    ])
    train_data = MyDataset(train_list, TensorTransform=TensorTransform)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, # 这里因为cifar100数据集自带随机性，所以不用随机打乱
                              num_workers=0)  # batch_size是从这里的DataLoader传递进去的
    valid_data = MyDataset(valid_list, TensorTransform=TensorTransform)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True,  # 这里因为cifar100数据集自带随机性，所以不用随机打乱
                              num_workers=0)  # batch_size是从这里的DataLoader传递进去的
    net = ViT(
        image_size=32,
        patch_size=16,
        num_classes=100,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2201,  # 参数量：53163514
        dropout=0.1,
        emb_dropout=0.1
    )
    if first_iter != 0:
        net.load_state_dict(torch.load(f'{save_path}/{first_iter}.pth'))
    print('网络参数量：', sum([param.nelement() for param in net.parameters()]))
    net.train() # 开启dropout模式
    ViT_processor = nn_processor(train_loader,valid_loader=valid_loader)
    # test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=None, download=False)
    ViT_processor.train(net,lr=lr, EPOCH=Epoch, max_iter=max_iter,
                        print_iter=print_iter,save_path= save_path,first_iter=first_iter,loss_func=loss_func,
                        test_dataset=False,roll=False,save_iter=save_iter,weight_decay=L2,lr_beta=lr_beta)
    # torch.save(net.state_dict(), f'{save_path}/final.pth')

