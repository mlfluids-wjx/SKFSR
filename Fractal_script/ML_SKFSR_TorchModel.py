import torch
import torch.nn as nn
import torch.nn.functional as F

class MRConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7]):
        super(MRConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding='same')
            for k in kernel_sizes
        ])
        # self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        # outputs = [self.bn(conv(x)) for conv in self.convs]
        # return outputs[0] + outputs[1] + outputs[2]
        return torch.cat(outputs, axis=1)
    
# class MMRConv(nn.Module):
#     def __init__(self, in_channels, out_channels, k=[3,5,7]):
#         super(MMRConv, self).__init__()
#         self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=k[1], padding='same')
#         self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=k[2], padding='same')
#         self.conv3e = nn.Conv2d(in_channels*2, out_channels, kernel_size=k[1], padding='same')
#         self.conv5e = nn.Conv2d(in_channels*2, out_channels, kernel_size=k[2], padding='same')
#         self.convf = nn.Conv2d(out_channels*2, out_channels, kernel_size=k[0], padding='same')
#         self.t = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         x3 = self.conv3(x)
#         x3 = self.t(x3)
#         x5 = self.conv5(x)
#         x5 = self.t(x5)
#         x35 = torch.cat([x3,x5], axis=1)
#         x33 = self.conv3e(x35)
#         x33 = self.t(x33)
#         x55 = self.conv5e(x35)
#         x55 = self.t(x55)
#         xf = torch.cat([x33,x55], axis=1)
#         xf = self.convf(xf)
#         xf = self.t(xf)
#         return xf

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes=[3,5]):
#         super(DoubleConv, self).__init__()
#         self.convs = nn.ModuleList([
#             nn.Conv2d(in_channels, out_channels, kernel_size=k, padding='same')
#             for k in kernel_sizes
#         ])
#         self.t = nn.ReLU(True)
        
#     def forward(self, x):
#         outputs = [conv(x) for conv in self.convs]
#         return self.t(outputs[0] + outputs[1])

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding='same'),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding='same'),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_conv(x2)
        
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # x = torch.cat([x2, x1], dim=1)
        return x2


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x


class model(nn.Module):
    def __init__(self, scaler=1):
        super(model, self).__init__()
        self.ls = [32, 64, 128, 256, 512]
        self.scaler = scaler
        self.conv1 = DoubleConv(1, self.ls[0])
        self.conv2 = DoubleConv(self.ls[0], self.ls[0])
        self.conv3 = DoubleConv(self.ls[1], self.ls[1])
        self.conv4 = DoubleConv(self.ls[2], self.ls[2])
        self.conv5 = DoubleConv(self.ls[3], self.ls[3])
        self.conv6 = DoubleConv(self.ls[4], self.ls[3])
        self.conv7 = DoubleConv(self.ls[4], self.ls[3])
        self.conv8 = DoubleConv(self.ls[3]+self.ls[1], self.ls[2])
        self.conv9 = DoubleConv(self.ls[2]+self.ls[0], self.ls[1])
        self.conv10 = DoubleConv(self.ls[1], self.ls[0])
        
        self.down1 = DownLayer(self.ls[0], self.ls[1])
        self.down2 = DownLayer(self.ls[1], self.ls[2])
        self.down3 = DownLayer(self.ls[2], self.ls[3])
        self.down4 = DownLayer(self.ls[3], self.ls[4])
        
        self.up1 = Up(self.ls[4], self.ls[3])
        self.up2 = Up(self.ls[3], self.ls[2])
        self.up3 = Up(self.ls[3], self.ls[2])
        self.up4 = Up(self.ls[2], self.ls[1])
        
        # self.convt2 = nn.ConvTranspose2d(self.ls[0], self.ls[0], kernel_size=2, stride=2)
        # self.convt3 = nn.ConvTranspose2d(self.ls[1], self.ls[1], kernel_size=2, stride=2)
        # self.convt4 = nn.ConvTranspose2d(self.ls[2], self.ls[2], kernel_size=2, stride=2)
        # self.convt5 = nn.ConvTranspose2d(self.ls[3], self.ls[3], kernel_size=2, stride=2)
        
        self.up_conv = nn.ConvTranspose2d(self.ls[1], self.ls[0], kernel_size=2, stride=2)
        
        # self.mmr = MMRConv(self.ls[0], self.ls[0], [3, 5, 7])
        self.last_conv = MRConv(self.ls[0]//(self.scaler)**2, self.ls[0], [3, 5, 7])
        self.econv = DoubleConv(self.ls[0]*3, self.ls[0]*2)
        # self.econv2 = DoubleConv(self.ls[0]*2, self.ls[0]*2)
        self.endconv = nn.Conv2d(self.ls[0]*2, 1, 3, padding='same')
        self.ps = nn.PixelShuffle(self.scaler)
    
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.upsamp = nn.Upsample(scale_factor=(2,2))
        self.t = nn.ReLU(inplace=True)
    
    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x1 = self.conv1(x)
        x1u = self.upsamp(x1)
        # x1u = self.conv2(x1u)
        # x1u = self.convt2(x1)
        x2 = self.down1(x1)
        x2u = self.upsamp(x2)
        # x2u = self.conv3(x2u)
        # x2u = self.convt3(x2)
        x3 = self.down2(x2)
        x3u = self.upsamp(x3)
        # x3u = self.conv4(x3u)
        # x3u = self.convt4(x3)
        x4 = self.down3(x3)
        x4u = self.upsamp(x4)
        # x4u = self.conv5(x4u)
        # x4u = self.convt5(x4)
        x5 = self.down4(x4)
        
        x1_up = self.up1(x4, x5)
        x1_up = torch.cat([x4, x1_up], axis=1)
        x1_up = self.conv6(x1_up)
        
        x2_up = self.up2(x3, x1_up)
        x2_up = torch.cat([x3, x2_up, x4u], axis=1)
        x2_up = self.conv7(x2_up)
        
        x3_up = self.up3(x2, x2_up)
        x3_up = torch.cat([x2, x3_up, x3u], axis=1)
        x3_up = self.conv8(x3_up)
        
        x4_up = self.up4(x1, x3_up)
        x4_up = torch.cat([x1, x4_up, x2u], axis=1)
        x4_up = self.conv9(x4_up)
       
        x5_up = self.up_conv(x4_up)
        x5_up = self.t(x5_up)
        x5_up = torch.cat([x1u, x5_up], axis=1)
        x5_up = self.conv10(x5_up)
       
        if self.scaler > 1:
            x5_up = self.ps(x5_up) 
       
        # x5_up = self.last_conv(x5_up)
        x6_up = self.last_conv(x5_up)
        x6_up = self.t(x6_up)
        
        x7_up = self.econv(x6_up)
        output = self.endconv(x7_up)
        return output

# model = model(scaler=4)
# a = torch.rand(1,1,512,256)
# b = model(a)


  

                