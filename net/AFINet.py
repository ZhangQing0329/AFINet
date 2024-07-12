import torch
import torch.nn as nn
import torch.nn.functional as F
from net.ResNet import resnet50
from math import log
from net.Res2Net import res2net50_v1b_26w_4s

class MultiScale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MultiScale, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 3, padding=1),
        )
        self.branch1 = nn.Sequential(
            
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=2, dilation=2)
        )
        self.branch2 = nn.Sequential(
            
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=4, dilation=4)
        )

        self.conv_cat = BasicConv2d(4*out_channel, 4*out_channel, 3, padding=1)
        
        self.conv0 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.conv1 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.conv2 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.conv3 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        
    def forward(self, x, edge):
         
        edge = F.upsample(edge, size=x.size()[2:], mode='bilinear') 
        
        y = torch.chunk(x, 4, dim=1)
        f0_1 = self.conv0(y[0])
        f1 = self.branch0(f0_1 * edge + f0_1)
        
        f1_1 = self.conv1(y[1] + f1)        
        f2 = self.branch1(f1_1 * edge + f1_1)
        
        f2_1 = self.conv2(y[2] + f2) 
        f3 = self.branch2(f2_1 * edge + f2_1)
        
        f3_1 = self.conv3(y[3] + f3)       
        f4 = self.branch3(f3_1 * edge + f3_1)
        
        f = self.conv_cat(torch.cat([f1, f2, f3, f4], dim=1)) + x
        
        return f

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class feature_exctraction(nn.Module):
    def __init__(self, in_channel, depth, kernel):
        super(feature_exctraction, self).__init__()
        self.in_channel = in_channel
        self.depth = depth
        self.kernel = kernel
        self.conv1 = nn.Sequential(nn.Conv2d(self.depth, self.depth, self.kernel, 1, (self.kernel - 1) // 2),
                                   nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.in_channel, self.depth, 3, 1, 1), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))

    def forward(self, x):

        output = self.conv3(x)

        return output

    #def initialize(self):
     #   weight_init(self)

class SANet(nn.Module):

    def __init__(self, in_dim, coff):
        super(SANet, self).__init__()
        self.dim = in_dim
        self.coff = coff
        self.k = 9
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 4, (1, self.k), 1, (0, self.k // 2)), nn.BatchNorm2d(self.dim // 4), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 4, (self.k, 1), 1, (self.k // 2, 0)), nn.BatchNorm2d(self.dim // 4), nn.ReLU(inplace=True))
        self.conv2_1 = nn.Conv2d(self.dim // 4, 1, (self.k, 1), 1, (self.k // 2, 0))
        self.conv2_2 = nn.Conv2d(self.dim // 4, 1, (1, self.k), 1, (0, self.k // 2))
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(x)
        conv2_1 = self.conv2_1(conv1_1)
        conv2_2 = self.conv2_2(conv1_2)
        conv3 = torch.add(conv2_1, conv2_2)
        conv4 = torch.sigmoid(conv3)

        conv5 = conv4.repeat(1, self.dim // self.coff, 1, 1)

        return conv5

    #def initialize(self):
     #   weight_init(self)

class SENet(nn.Module):

    def __init__(self, in_dim, ratio=2):
        super(SENet, self).__init__()
        self.dim = in_dim
        self.fc = nn.Sequential(nn.Linear(in_dim, self.dim // ratio, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // ratio, in_dim, bias = False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y=F.adaptive_avg_pool2d(x, (1,1)).view(b,c)
        y= self.sigmoid(self.fc(y)).view(b,c,1,1)
 
        output = y.expand_as(x)

        return output
        
class CANet(nn.Module):

    def __init__(self, in_dim):
        super(CANet, self).__init__()
        self.dim = in_dim
        self.fc = nn.Sequential(nn.Linear(in_dim, self.dim // 16, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // 16, in_dim-1, bias = False))
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.max_pool(x).view(b,c)

        output = self.sigmoid(self.fc(y)).view(b,64,1,1)

 
        return output

    #def initialize(self):
    #    weight_init(self)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2,
                              bias=False)  # infer a one-channel attention map

    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True)  # [B, 1, H, W], average
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True)  # [B, 1, H, W], max
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1)  # [B, 2, H, W]
        att_map = torch.sigmoid(self.conv(ftr_cat))  # [B, 1, H, W]
        return att_map

    #def initialize(self):
    #    weight_init(self)

class Edge_detect(nn.Module):

    def __init__(self):
        super(Edge_detect, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        output = self.conv3(self.relu(conv2 + x))

        return output

class Fusion_3(nn.Module):

    def __init__(self):
        super(Fusion_3, self).__init__()
        self.se0 = SENet(64)
        self.se1 = SENet(64)
        self.se2 = SENet(64)
        
        self.conv1 = nn.Sequential(nn.Conv2d(64*3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        

    def forward(self, x0, x1, x2): # x1 and x2 are the adjacent features
        b, c, h, w = x0.size()
        x1_up = F.upsample(x1, size=x0.size()[2:], mode='bilinear')
        x2_up = F.upsample(x2, size=x0.size()[2:], mode='bilinear')
        x0_w = self.se0(x0)
        x1_w = self.se1(x1_up)
        x2_w = self.se2(x2_up)
        se = x0_w * x1_w * x2_w
        w0 = x0_w + se
        w1 = x1_w + se
        w2 = x2_w + se
        x0_ref = x0 * w0
        x1_ref = x1_up * w1
        x2_ref = x2_up * w2
        output = self.conv1(torch.cat([x0_ref, x1_ref, x2_ref], dim=1))+ x0
        
        return output, x1_ref
        
class Fusion_2(nn.Module):

    def __init__(self):
        super(Fusion_2, self).__init__()
        self.se0 = SENet(64)
        self.se1 = SENet(64)
        
        self.conv1 = nn.Sequential(nn.Conv2d(64*2, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        

    def forward(self, x0, x1): # x1 is the adjacent features
        x1_up = F.upsample(x1, size=x0.size()[2:], mode='bilinear')
        x0_w = self.se0(x0)
        x1_w = self.se1(x1_up)
        se = x0_w * x1_w
        w0 = x0_w + se
        w1 = x1_w + se
        x0_ref = x0 * w0
        x1_ref = x1_up * w1
        output = self.conv1(torch.cat([x0_ref, x1_ref], dim=1)) + x0
        
        return output
        
class ResidualLearning(nn.Module):

    def __init__(self, channel=64):
        super(ResidualLearning, self).__init__()
               
        self.decoder6 = nn.Sequential(
            BasicConv2d(channel, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            #nn.Dropout(0.5),
        )
        self.S6 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.up6 =  TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        
        self.decoder5 = nn.Sequential(
            BasicConv2d(channel, 64, 3, padding=1), 
            BasicConv2d(64, 64, 3, padding=1))
            #nn.Dropout(0.5),
        self.SA5 =  SENet(64, 2)
        self.up5 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
       
        self.S5 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        
        self.decoder4 = nn.Sequential(
            BasicConv2d(channel*2, 64, 3, padding=1), 
            BasicConv2d(64, 64, 3, padding=1))
            #nn.Dropout(0.5),
        self.SA4 =  SENet(64, 2)
        self.up4 =  TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        
        self.S4 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        
        self.decoder3 = nn.Sequential(
            BasicConv2d(channel*2, 64, 3, padding=1), 
            BasicConv2d(64, 64, 3, padding=1))
            #nn.Dropout(0.5),
        self.SA3 =  SENet(64, 2)
        self.up3 =  TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        
        self.S3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)  
        
        self.decoder2 = nn.Sequential(
            BasicConv2d(channel*2, 64, 3, padding=1), 
            BasicConv2d(64, 64, 3, padding=1))
            #nn.Dropout(0.5),
        self.SA2 =  SENet(64, 2)  
        self.up2 =  TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)      
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1) 
        
        self.decoder1 = nn.Sequential(
            BasicConv2d(channel*2, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1))
        self.SA1 =  SENet(64, 2)
        self.up1 =  TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        
        self.S1 = nn.Conv2d(64, 1, 3, stride=1, padding=1) 
        
        self.conv5 = BasicConv2d(64, 64, 3, padding=1)
        self.conv4 = BasicConv2d(64, 64, 3, padding=1) 
        self.conv3 = BasicConv2d(64, 64, 3, padding=1) 
        self.conv2 = BasicConv2d(64, 64, 3, padding=1) 
        self.edgefusion5 = MultiScale(16, 16) 
        self.edgefusion4 = MultiScale(16, 16) 
        self.edgefusion3 = MultiScale(16, 16) 
        self.edgefusion2 = MultiScale(16, 16)                    

    def forward(self, x2, x3, x4, x5, edge):  # x1 is the saliency map from the adjacent deep layer
               
        #x6_ref = self.decoder6(x6)
        
        
        #x6_up = F.upsample(x6_up, size=x5.size()[2:], mode='bilinear')
        #edge_up5 = F.upsample(edge, size=x5.size()[2:], mode='bilinear')
        x5_cat = self.edgefusion5(x5, edge)
        #x5_ref = self.SA5(x5_cat) * x5_cat
        x5_ref = self.decoder5(x5_cat)
        x5_up = self.up5(x5_ref)
        s5 = self.S5(x5_ref)
        
        #edge_up4 = F.upsample(edge, size=x4.size()[2:], mode='bilinear')
        x4_edge = self.edgefusion4(x4, edge)
        x4_cat = torch.cat((x4_edge, x5_up), 1)
        x4_ref = self.decoder4(x4_cat)
        #x4_ref = self.SA4(x4_cat) * x4_cat
        x4_up = self.up4(x4_ref)
        s4 = self.S4(x4_ref)
        
        #edge_up3 = F.upsample(edge, size=x3.size()[2:], mode='bilinear')
        x3_edge = self.edgefusion3(x3, edge)
        x3_cat = torch.cat((x3_edge, x4_up), 1)
        x3_ref = self.decoder3(x3_cat)
        #x3_ref = self.SA3(x3_cat) * x3_cat
        x3_up = self.up3(x3_ref)
        s3 = self.S3(x3_ref)
        
        #edge_up2 = F.upsample(edge, size=x2.size()[2:], mode='bilinear')
        x2_edge = self.edgefusion2(x2, edge)
        x2_cat = torch.cat((x2_edge, x3_up), 1)
        x2_ref = self.decoder2(x2_cat)
        #x2_ref = self.SA2(x2_cat) * x2_cat

        s2 = self.S2(x2_ref)
                                
        return s2, s3, s4, s5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # if self.training:
        # self.initialize_weights()

        self.fusion4 = Fusion_3()
        self.fusion3 = Fusion_3()
        self.fusion2 = Fusion_2()
        self.fusion5 = Fusion_2()
        
        self.fem_layer5 = feature_exctraction(64, 64, 3)
        self.fem_layer4 = feature_exctraction(64, 64, 3)
        self.fem_layer3 = feature_exctraction(64, 64, 3)
        self.fem_layer2 = feature_exctraction(64, 64, 3)
        
        self.e_layer5 = feature_exctraction(64, 64, 3)
        self.e_layer2 = feature_exctraction(64, 64, 3)
        self.e_conv = nn.Sequential(nn.Conv2d(64*2, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 1, 1, 1, 0))

        self.con2 = nn.Sequential(nn.Conv2d(256, 64, 1, 1, 0),nn.BatchNorm2d(64),nn.ReLU(inplace = True))
        self.con3 = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0),nn.BatchNorm2d(64),nn.ReLU(inplace = True))
        self.con4 = nn.Sequential(nn.Conv2d(1024, 64, 1, 1, 0),nn.BatchNorm2d(64),nn.ReLU(inplace = True))
        self.con5 = nn.Sequential(nn.Conv2d(2048, 64, 1, 1, 0),nn.BatchNorm2d(64),nn.ReLU(inplace = True))
        
        
        
        self.res1 = ResidualLearning()

        self.conv1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.conv2 = nn.Sequential(nn.Conv2d(65, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 1, 1, 1, 0))
        self.pooling = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(2048, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 64, 7, 1, 3), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 64, 7, 1, 3), nn.BatchNorm2d(64), nn.ReLU(inplace =True))
        self.conv4 = nn.Conv2d(64, 1, 1, 1, 0)
        self.s6 = nn.Conv2d(64, 1, 1, 1, 0)

    # def initialize_weights(self):
    # model_state = torch.load('./models/resnet50-19c8e357.pth')
    # self.resnet.load_state_dict(model_state, strict=False)

    def forward(self, x):
        conv2, conv3, conv4, conv5 = self.resnet(x)
        
        x_size = x.size() 
        
        feature6 = self.conv3(self.pooling(conv5))
        s6 = self.s6(feature6)     

        conv2 = self.con2(conv2)
        conv3 = self.con3(conv3)
        conv4 = self.con4(conv4)
        conv5 = self.con5(conv5)

        fea_layer5 = self.fem_layer5(conv5)
        fem_layer5 = F.upsample(feature6, size=fea_layer5.size()[2:], mode='bilinear') * fea_layer5 + fea_layer5
        fea_layer4 = self.fem_layer4(conv4)
        fem_layer4 = F.upsample(feature6, size=fea_layer4.size()[2:], mode='bilinear') * fea_layer4 + fea_layer4
        fea_layer3 = self.fem_layer3(conv3)
        fem_layer3 = F.upsample(feature6, size=fea_layer3.size()[2:], mode='bilinear') * fea_layer3 + fea_layer3
        fea_layer2 = self.fem_layer2(conv2)
        fem_layer2 = F.upsample(feature6, size=fea_layer2.size()[2:], mode='bilinear') * fea_layer2 + fea_layer2
        
        e_layer5 = self.e_layer5(conv5)
        e_layer2 = self.e_layer2(conv2)
        e_layer5_up = F.upsample(e_layer5, size=e_layer2.size()[2:], mode='bilinear')
        edge = torch.sigmoid(self.e_conv(torch.cat([e_layer5_up, e_layer2], dim=1)))

        
        fea4, att4 = self.fusion4(fem_layer4, fem_layer3, fem_layer5)
        fea3, att3 = self.fusion3(fem_layer3, fem_layer2, fem_layer4)
        fea2 = self.fusion2(fem_layer2, fem_layer3)
        fea5 = self.fusion5(fem_layer5, fem_layer4)
        
        s2, s3, s4, s5 = self.res1(fea2, fea3, fea4, fea5, edge)
        
        pre6 = F.upsample(edge, size=x.size()[2:], mode='bilinear')
        pre5 = F.upsample(s6, size=x.size()[2:], mode='bilinear')
        pre4 = F.upsample(s5, size=x.size()[2:], mode='bilinear')
        pre3 = F.upsample(s4, size=x.size()[2:], mode='bilinear')
        pre2 = F.upsample(s3, size=x.size()[2:], mode='bilinear')
        pre1 = F.upsample(s2, size=x.size()[2:], mode='bilinear')


        return pre1, pre2, pre3, pre4, pre5, pre6
        
