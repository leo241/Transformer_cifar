import torch
from vit_pytorch import ViT
# 创建ViT模型实例
# net = ViT(
#     image_size = 32,
#     patch_size = 4,
#     num_classes = 100,
#     dim = 2056,
#     depth = 6,
#     heads = 3,
#     mlp_dim = 1749, # 参数量：53163514
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
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
# 随机化一个图像输入
x = torch.randn(1, 3, 32, 32)
# 获取输出
output = net(x)
print(output.shape)
# 查看网络参数量，方法一
print('网络参数量：',sum([param.nelement() for param in net.parameters()]))
torch.save(net.state_dict(), f'model_save/0.pth')