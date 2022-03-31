import torch
import torch.nn as nn
class Vgg(nn.Module):
    def __init__(self, config):
        super(Vgg, self).__init__()
        self.stage1 = self.make_layer(3, 64, [3 for i in range(config[1])], repeat=config[1])
        self.stage2 = self.make_layer(64, 128, [3 for i in range(config[2])], repeat=config[2])
        self.stage3 = self.make_layer(128, 256, [3 for i in range(config[3])], repeat=config[3])
        self.stage4 = self.make_layer(256, 512, [3 for i in range(config[4])], repeat=config[4])
        self.stage5 = self.make_layer(512, 512, [3 for i in range(config[5])], repeat=config[5])
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.5)
    def make_layer(self, in_channel, out_channel, kernel_size, repeat):
        layer = []
        for i in range(repeat):
            layer.append(nn.Conv2d(in_channel, out_channel, kernel_size[i], padding=1))
            layer.append(nn.BatchNorm2d(out_channel))
            layer.append(nn.ReLU(inplace=True))
            in_channel = out_channel
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
def vgg_11():
    return Vgg([-1, 1, 1, 2, 2, 2])

def vgg_13():
    return Vgg([-1, 1, 2, 2, 2, 2])

def vgg_16():
    return Vgg([-1, 1, 2, 2, 3, 3])

def vgg_19():
    return Vgg([-1, 2, 2, 4, 4, 4])
x = torch.rand((10, 3, 224, 224))
net = vgg_19()
print(net(x).shape)
'''import hiddenlayer as h
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
vis_graph = h.build_graph(net, torch.zeros([1, 3, 224, 224]))  # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
vis_graph.save("./demo1.png")'''