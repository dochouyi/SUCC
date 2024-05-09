import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(AttentionBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels//4, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels//4, output_channels//4, 3, stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels//4, output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
        self._initialize_weights()

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight,)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AttentionModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()

        self.trunk_branches = nn.Sequential(
            AttentionBlock(in_channels, out_channels),
            AttentionBlock(out_channels, out_channels)
        )
        self.mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax_blocks = AttentionBlock(in_channels, out_channels)
        self.interpolation = nn.UpsamplingBilinear2d(scale_factor=2)
        self.softmax2_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )
        self._initialize_weights()
    def forward(self, x):
        out_trunk = self.trunk_branches(x)
        out_mpool = self.mpool(x)
        out_softmax = self.softmax_blocks(out_mpool)
        out_interp = self.interpolation(out_softmax)
        out_softmax = self.softmax2_blocks(out_interp)
        out = (1 + out_softmax) * out_trunk

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight,)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)





