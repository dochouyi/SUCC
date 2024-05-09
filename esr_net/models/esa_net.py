import torch.nn as nn
from torchvision import models
from esr_net.models.attention_util import AttentionModule
import torch

class ESA_net(nn.Module):
    def __init__(self, load_weights=False,weight_path=''):
        super(ESA_net, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)
        self.AttentionModule_1=AttentionModule(512,256)
        self.AttentionModule_2=AttentionModule(256,256)
        self.backend_feat_den = [256, 128]
        self.backend_den = make_layers(self.backend_feat_den,in_channels = 256, dilation = True)
        self.output_den_map = nn.Conv2d(128, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())
        else:
            checkpoint = torch.load(weight_path)
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self,x):
        x = self.frontend(x)
        x = self.AttentionModule_1(x)
        x = self.AttentionModule_2(x)
        x = self.backend_den(x)
        x = self.output_den_map(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight,)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels = 3, batch_norm=False, dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                
