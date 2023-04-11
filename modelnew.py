import pdb
import torch.nn.functional as F
import torch.nn as nn
import torch
class SCM(torch.nn.Module):
    def __init__(self,input_size, output_size):
        super(SCM,self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.Relu = nn.LeakyReLU(0.2, True)
        self.conv_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        x1,x2=x.chunk(2,dim=1)
        x_up=F.sigmoid(self.conv(x1))
        x_down=F.sigmoid(self.conv_1(x2))
        out=torch.cat((x1*x_up,x2*x_down),1)

        return out
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_size, output_size, kernel_size, stride, 0),
            nn.LeakyReLU(0.2, True)
        )
    def forward(self, x):
        return self.conv(x)

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_size, output_size*16, kernel_size, 1, 0),
            nn.PixelShuffle(4)
        )
    def forward(self, x):
        out=self.deconv(x)
        return out

class UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.up_conv3 = DeconvBlock(num_filter, num_filter)
        self.SCM1 = SCM(num_filter, num_filter)
        self.SCM2 = SCM(num_filter, num_filter)
    def forward(self, x):
        h0_1 = self.SCM1(self.up_conv1(x))
        l0 = self.up_conv2(h0_1)
        h1 = self.SCM2(self.up_conv3(l0-x))
        return h1 + h0_1

class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.down_conv2 = DeconvBlock(num_filter, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.SCM = SCM(num_filter, num_filter)
    def forward(self, x):

        l0_1 = self.down_conv1(x)

        h0_1 = self.SCM(self.down_conv2(l0_1))
        l1 = self.down_conv3(h0_1 - x)
        return l1 + l0_1

class downMLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(downMLP, self).__init__()
        self.linear = nn.Conv1d(input_size, common_size, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.linear(x)

class upMLP(nn.Module):

    def __init__(self, input_size, common_size):
        super(upMLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Conv1d(input_size, common_size, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.linear(x)

class Channelup(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Channelup, self).__init__()
        self.up_MLP1 = upMLP(in_channels, out_channels)
        self.up_MLP2 = upMLP(in_channels, out_channels)
        self.up_MLP3 = upMLP(in_channels, out_channels)

    def forward(self, feature):
        return (self.up_MLP1(feature)+self.up_MLP2(feature)+self.up_MLP3(feature)) / 3

class ChannelUpBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ChannelUpBlock, self).__init__()
        self.up_conv1 = Channelup(in_channels,out_channels)
        self.up_conv2 = downMLP(out_channels, in_channels)
        self.up_conv3 = Channelup(in_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        input = x.view(b, c, h * w)

        h0 = self.up_conv1(input)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - input)
        output = (h1 + h0).view(b, -1, h, w)
        return output

class ChannelDownBlock(torch.nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ChannelDownBlock, self).__init__()
        self.down_conv1 = downMLP(in_channels, out_channels)
        self.down_conv2 = Channelup(out_channels, in_channels)
        self.down_conv3 = downMLP(in_channels, out_channels)
        
    def forward(self, x):
        b, c, h, w = x.size()
        input = x.view(b, c, h * w)

        l0 = self.down_conv1(input)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - input)
        output = (l1 + l0).view(b, -1, h, w)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.HRconv1=ConvBlock(1,32,3,1)
        self.HRup=ChannelUpBlock(32,128)
        self.HRdown1 = ChannelDownBlock(128, 32)
        self.HRup1 = ChannelUpBlock(32, 128)
        self.LRconv1 = ConvBlock(4*2, 128, 3, 1)
        self.LRup= UpBlock(128)
        self.LRdown1 = DownBlock(128)
        self.LRup1=UpBlock(128)
        self.finall=nn.Sequential(
            nn.Conv2d(128*3, 128, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 64, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 32, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 4*2, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self,ms,pan):

        F1 = self.HRup1(self.HRdown1(self.HRup(self.HRconv1(pan))))
        lrhs2=self.LRdown1(self.LRup(self.LRconv1(ms)))
        feature2=self.LRup1(lrhs2)

        out1=(F1+feature2)*0.5


        hrms3=self.HRdown1(out1)
        F3=self.HRup1(hrms3)
        lrhs3=self.LRdown1(out1)
        F4=self.LRup1(lrhs3)

        out2 =(F3 +F4)*0.5

        hrms4 = self.HRdown1(out2)
        F5= self.HRup1(hrms4)
        lrhs4 = self.LRdown1(out2)
        F6 = self.LRup1(lrhs4)

        out3 = (F5 + F6) * 0.5

        out=torch.cat((out1,out2,out3),1)

        return self.finall(out)
if __name__ == '__main__':


    tnt = Net()

    ms = torch.randn(1, 4, 64, 64)
    pan = torch.randn(1, 1, 256, 256)

    logits = tnt(ms,pan) # (2, 1000)
    print('ok')