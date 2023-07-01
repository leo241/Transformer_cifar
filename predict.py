import numpy as np
from vit_pytorch import ViT
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch
import torch.nn as nn
from tqdm import tqdm


def predict_all(net,test_dataset = False,train = False,length = 2500): # 给定验证集列表，得到预测的结果
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速
    net = net.to(device)
    if not test_dataset:
        test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=train, transform=None, download=False)
    x = test_dataset.data
    y = test_dataset.targets
    whole_length = len(x) # 整个数据集的长度，如果是train就是60000，如果是test就是10000
    id = 0
    cnt = 0
    while id < whole_length:
        x1,y1 = x[id:id + length],y[id:id + length]
        TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])
        vx = [TensorTransform(item).unsqueeze(0) for item in x1]  # 256,256 -> 1，1，256,256

        vx = torch.cat(vx, dim=0).to(device) # 这个位置记得加入GPU！一切预测的地方都要注意
        net.train() # 因为已经把网络dropout取0了，所以不用开启eval模式
        with torch.no_grad():
            ypre = net(vx)
            ypre = torch.argmax(ypre,dim=1)
            cnt += sum(np.asarray(y1) == np.asarray(ypre.cpu()))
        id += length # 滑动窗口
    accuracy = cnt / whole_length
    return round(accuracy,3)


if __name__ == '__main__':
    length = 2500 # 切片长度，防止一次性cpu装不下,应尽可能调大这个值
    model_root = 'model_save'

    print('常规预测')
    print('截断样本量：',length)
    print('-------------------------------------')

    # net = ViT(
    #     image_size=32,
    #     patch_size=4,
    #     num_classes=100,
    #     dim=2056,
    #     depth=6,
    #     heads=3,
    #     mlp_dim=1749,  # 参数量：53163514
    #     dropout=0,
    #     emb_dropout=0
    # )
    net = ViT(
        image_size=32,
        patch_size=16,
        num_classes=100,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2201,  # 参数量：53163514
        dropout=0,
        emb_dropout=0
    )
    # models = list(range(550000, 1410001, 2000))
    models = [2008000]
    for model in models:
        net.load_state_dict(torch.load(f'{model_root}/{model}.pth'))
        print(model, '训练集准确率:', predict_all(net, train=True, length=length),'测试集准确率:',predict_all(net,train=False,length=length))
        # train_writer = SummaryWriter('logs_train')
        # test_writer = SummaryWriter('logs_test')

