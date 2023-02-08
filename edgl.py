
import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DWConv, self).__init__()

        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.01, affine=True)

    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        out=self.bn(out)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BasicConv4(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv4, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class ChannelGate4(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate4, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(),
                                 nn.Linear(gate_channels, gate_channels // reduction_ratio),
                                 nn.ReLU(),
                                 nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x): #x:(b,c,d,d)
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_avg = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_max = self.mlp(max_pool)

        scale_avg = torch.sigmoid(channel_att_avg).unsqueeze(2).unsqueeze(3).expand_as(x) 
        scale_max = torch.sigmoid(channel_att_max).unsqueeze(2).unsqueeze(3).expand_as(x) 
        return x * scale_avg,x*scale_max  
class SpatialGate4(nn.Module):
    def __init__(self):
        super(SpatialGate4, self).__init__()
        kernel_size = 7
        self.spatial_avg = BasicConv4(3, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)  
        self.spatial_max = BasicConv4(3, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)  


    def forward(self, x_avg,x_max):
        x_compress_avg = torch.cat((torch.max(x_avg, 1)[0].unsqueeze(1), torch.mean(x_avg, 1).unsqueeze(1),torch.min(x_avg,1)[0].unsqueeze(1)), dim=1)
        x_compress_max = torch.cat((torch.max(x_max, 1)[0].unsqueeze(1), torch.mean(x_max, 1).unsqueeze(1),torch.min(x_max,1)[0].unsqueeze(1)), dim=1)

        x_out_avg= self.spatial_avg(x_compress_avg)
        x_out_max=self.spatial_max(x_compress_max)

        scale_avg = torch.sigmoid(x_out_avg)
        scale_max=torch.sigmoid(x_out_max)
      
        return x_avg*scale_avg,x_max*scale_max
class S2A2M_inner(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(S2A2M_inner, self).__init__()
        kernel_size = 7
        self.ChannelGate = ChannelGate4(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate4()
        self.fusion=DWConv(2*gate_channels, gate_channels)

    def forward(self, x):
        x_out_avg,x_out_max=self.ChannelGate(x)
        x_out_avg,x_out_max = self.SpatialGate(x_out_avg,x_out_max)
        x_out=self.fusion(torch.cat((x_out_avg,x_out_max),dim=1))

        return x_out

class BasicBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class UM3(nn.Module):  
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(UM3, self).__init__()
        norm_layer = nn.BatchNorm2d
        scale_width = int(planes / 4)

        self.scale_width = scale_width

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)

        self.conv1_1 = conv3x3(scale_width, scale_width)
        self.bn1_1 = norm_layer(scale_width)
        self.conv1_2 = conv3x3(scale_width, scale_width)
        self.bn1_2=norm_layer(scale_width)
        self.conv1_3 = conv3x3(scale_width, scale_width)
        self.bn1_3 = norm_layer(scale_width)
        self.conv1_4 = conv3x3(scale_width, scale_width)
        self.bn1_4=norm_layer(scale_width)
       
        self.conv2_1 = conv3x3(scale_width, scale_width)
        self.bn2_1 = norm_layer(scale_width)
        self.conv2_2 = conv3x3(scale_width, scale_width)
        self.bn2_2=norm_layer(scale_width)
        self.conv2_3 = conv3x3(scale_width, scale_width)
        self.bn2_3 = norm_layer(scale_width)
        self.conv2_4 = conv3x3(scale_width, scale_width)
        self.bn2_4=norm_layer(scale_width)
        self.conv2_5 = conv3x3(scale_width, scale_width)
        self.bn2_5 = norm_layer(scale_width)
        self.conv2_6 = conv3x3(scale_width, scale_width)
        self.bn2_6=norm_layer(scale_width)

        self.conv3_1 = conv3x3(scale_width, scale_width)
        self.bn3_1 = norm_layer(scale_width)
        self.conv3_2 = conv3x3(scale_width, scale_width)
        self.bn3_2=norm_layer(scale_width)
        self.conv3_3 = conv3x3(scale_width, scale_width)
        self.bn3_3 = norm_layer(scale_width)
        self.conv3_4 = conv3x3(scale_width, scale_width)
        self.bn3_4=norm_layer(scale_width)

        self.conv4_1 = conv3x3(scale_width, scale_width)
        self.bn4_1 = norm_layer(scale_width)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        sp_x = torch.split(out, self.scale_width, 1)

        out1_1=self.conv1_1(sp_x[0])
        out1_1=self.bn1_1(out1_1)
        out1_2=self.conv1_2(sp_x[1])
        out1_2=self.bn1_2(out1_2)
        out1_3=self.conv1_3(sp_x[2])
        out1_3=self.bn1_3(out1_3)
        out1_4=self.conv1_4(sp_x[3])
        out1_4=self.bn1_4(out1_4)
        out1=out1_1+out1_2+out1_3+out1_4


        out2_1=self.conv2_1(self.relu(out1_1)+self.relu(out1_2))
        out2_1=self.bn2_1(out2_1)
        out2_2=self.conv2_2(self.relu(out1_1)+self.relu(out1_3))
        out2_2=self.bn2_2(out2_2)
        out2_3=self.conv2_3(self.relu(out1_1)+self.relu(out1_4))
        out2_3=self.bn2_3(out2_3)
        out2_4=self.conv2_4(self.relu(out1_2)+self.relu(out1_3))
        out2_4=self.bn2_4(out2_4)
        out2_5=self.conv2_5(self.relu(out1_2)+self.relu(out1_4))
        out2_5=self.bn2_5(out2_5)
        out2_6=self.conv2_6(self.relu(out1_3)+self.relu(out1_4))
        out2_6=self.bn2_6(out2_6)
        out2=out2_1+out2_2+out2_3+out2_4+out2_5+out2_6


        out3_1=self.conv3_1(self.relu(out1_1)+self.relu(out1_2)+self.relu(out1_3))
        out3_1=self.bn3_1(out3_1)
        out3_2=self.conv3_2(self.relu(out1_1)+self.relu(out1_2)+self.relu(out1_4))
        out3_2=self.bn3_2(out3_2)
        out3_3=self.conv3_3(self.relu(out1_1)+self.relu(out1_3)+self.relu(out1_4))
        out3_3=self.bn3_3(out3_3)
        out3_4=self.conv3_4(self.relu(out1_2)+self.relu(out1_3)+self.relu(out1_4))
        out3_4=self.bn3_4(out3_4)
        out3=out3_1+out3_2+out3_3+out3_4


        out4_1=self.conv4_1(self.relu(out1_1)+self.relu(out1_2)+self.relu(out1_3)+self.relu(out1_4))
        out4_1=self.bn4_1(out4_1)
        out4=out4_1
       
        out = torch.cat([out1, out2, out3, out4], dim=1)

       
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class S2A2M(nn.Module):   
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(S2A2M, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = S2A2M_inner(planes, 16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)  

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class CoreBlock(nn.Module):   
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CoreBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = S2A2M_inner(planes, 16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)  

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class EDGL_FLP(nn.Module):   

    def __init__(self, block_b, block_m, block_a,block_c, layers, num_classes=7):
        super(EDGL_FLP, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_b, 64, 64, layers[0])
        self.layer2 = self._make_layer(block_b, 64, 128, layers[1], stride=2)

        # In this branch, each BasicBlock replaced by AttentiveBlock.
        self.layer3_1_p1 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p1 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        self.layer3_1_p2 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p2 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        self.layer3_1_p3 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p3 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        self.layer3_1_p4 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p4 = self._make_layer(block_a, 256, 512, layers[3], stride=1)

        # In this branch, each BasicBlock replaced by MulScaleBlock.
        self.layer3_2 = self._make_layer(block_m, 128, 256, layers[2], stride=2)
        self.layer4_2 = self._make_layer(block_m, 256, 512, layers[3], stride=2)

        # In this branch, each BasicBlock replaced by CoreBlock.
        self.layer3_3 = self._make_layer(block_c, 128, 256, layers[2], stride=2)
        self.layer4_3 = self._make_layer(block_c, 256, 512, layers[3], stride=1)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_1 = nn.Linear(512, num_classes)
        self.fc_fuse_core=nn.Linear(1024,512)
        self.fc_2 = nn.Linear(512, num_classes)
        self.fc_p1p3=nn.Linear(1024,num_classes)
        self.fc_p2p4=nn.Linear(1024,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # branch local ############################################
        patch_11 = x[:, :, 0:14, 0:14]
        patch_12 = x[:, :, 0:14, 14:28]
        patch_21 = x[:, :, 14:28, 0:14]
        patch_22 = x[:, :, 14:28, 14:28]
        patch_core=x[:,:,2:26,2:26]

        branch_1_p1_out = self.layer3_1_p1(patch_11)
        branch_1_p1_out = self.layer4_1_p1(branch_1_p1_out)

        branch_1_p2_out = self.layer3_1_p2(patch_12)
        branch_1_p2_out = self.layer4_1_p2(branch_1_p2_out)

        branch_1_p3_out = self.layer3_1_p3(patch_21)
        branch_1_p3_out = self.layer4_1_p3(branch_1_p3_out)

        branch_1_p4_out = self.layer3_1_p4(patch_22)
        branch_1_p4_out = self.layer4_1_p4(branch_1_p4_out)

        branch_1_out_p2_p4=self.avgpool(branch_1_p2_out)+self.avgpool(branch_1_p4_out)
        branch_1_out_p2_p4=torch.flatten(branch_1_out_p2_p4/2,1)
        branch_1_out_p1_p3=self.avgpool(branch_1_p1_out)+self.avgpool(branch_1_p3_out)
        branch_1_out_p1_p3=torch.flatten(branch_1_out_p1_p3/2,1)

        branch_1_out_1 = torch.cat([branch_1_p1_out, branch_1_p2_out], dim=3)
        branch_1_out_2 = torch.cat([branch_1_p3_out, branch_1_p4_out], dim=3)
        branch_1_out = torch.cat([branch_1_out_1, branch_1_out_2], dim=2)

        branch_1_out = self.avgpool(branch_1_out)
        branch_1_out = torch.flatten(branch_1_out, 1)
        branch_1_out = self.fc_1(branch_1_out)

       
        

        # branch global ############################################
        branch_2_out = self.layer3_2(x)
        branch_2_out = self.layer4_2(branch_2_out)
        branch_2_out = self.avgpool(branch_2_out)
        branch_2_out = torch.flatten(branch_2_out, 1)
    

         # branch global cfa ############################################
        branch_3_out = self.layer3_3(patch_core)
        branch_3_out = self.layer4_3(branch_3_out)
        branch_3_out=self.avgpool(branch_3_out)
        branch_3_out = torch.flatten(branch_3_out, 1)

        branch_2_out_before=self.fc_fuse_core(torch.cat((branch_2_out,branch_3_out),dim=1))
        branch_2_out = self.fc_2(branch_2_out_before)

        branch_multi_p1_p3_out=torch.cat((branch_2_out_before,branch_1_out_p1_p3),dim=1)
        branch_multi_p1_p3_out=self.fc_p1p3(branch_multi_p1_p3_out)  
        branch_multi_p2_p4_out=torch.cat((branch_2_out_before,branch_1_out_p2_p4),dim=1)
        branch_multi_p2_p4_out=self.fc_p2p4(branch_multi_p2_p4_out)


        return branch_1_out, branch_2_out,branch_multi_p1_p3_out,branch_multi_p2_p4_out

    def forward(self, x):
        return self._forward_impl(x)


def edgl():
    return EDGL_FLP(block_b=BasicBlock, block_m=UM3, block_a=S2A2M, block_c=CoreBlock,layers=[2, 2, 2, 2])
