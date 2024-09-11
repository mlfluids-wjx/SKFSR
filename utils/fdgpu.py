import torch
from torch import nn

class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

class binarize(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x, threshold=None):
        if threshold == None:
            threshold = x.mean()
        xx = torch.subtract(x, threshold)
        xx = LBSign.apply(xx)
        xx = torch.add(xx, torch.ones_like(xx))
        xx = LBSign.apply(xx)
        return xx
    
class binarizen(nn.Module):
    def __init__(self, e=-50, **kwargs):
        super().__init__()
        self.e = e

    def forward(self, x, threshold=None):
        if threshold == None:
            threshold = x.mean()
        a = 1 / (1 + torch.exp(self.e * (x - threshold)))
        # a = nn.functional.sigmoid(x - threshold)
        # a = SigSign.apply(x - threshold)
        return a

class fd2df(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x, d):
        # x = torch.transpose(x, 0, 1)
        # with torch.no_grad():
        #     # px = int(np.ceil(x.shape[1]/d) * d - x.shape[1])
        #     py = int(np.ceil(x.shape[2]/d) * d - x.shape[2])
        #     pz = int(np.ceil(x.shape[3]/d) * d - x.shape[3])
        # x = pad(x, (0,pz,0,py,0,0), value=0)
        temp = nn.MaxPool2d(kernel_size=(d,d))(x)
        # temp = nn.MaxPool3d(kernel_size=(1,d,d))(x) 
        return temp
    
class fd2da(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x, d):
        # x = torch.transpose(x, 0, 1)
        # temp = nn.AvgPool3d(kernel_size=(1,d,d))(x) 
        temp = nn.AvgPool2d(kernel_size=(d,d))(x)
        return temp
    
