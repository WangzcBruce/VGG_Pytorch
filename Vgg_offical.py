import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
net = models.vgg11(pretrained=True)
#summary(net, input_size=[(3, 256, 256)], batch_size=2, device="cuda")
from torchviz import make_dot
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
x = torch.randn(1, 3, 224, 224).requires_grad_(True)  # 定义一个网络的输入值
y = net(x)  # 获取网络的预测值
print(net)
MyConvNetVis = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
MyConvNetVis.format = "png"
# 指定文件生成的文件夹
MyConvNetVis.directory = "data"
# 生成文件
MyConvNetVis.view()
import hiddenlayer as h
vis_graph = h.build_graph(net, torch.zeros([1, 3, 224, 224]))  # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
vis_graph.save("./demo1.png")