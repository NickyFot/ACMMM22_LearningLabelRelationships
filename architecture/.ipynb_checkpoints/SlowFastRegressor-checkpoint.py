import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.cuda.amp import autocast

from .slowfast import slowfast101

class Regressor(nn.Module):
    def __init__(self, outdim: int, variance: bool = True, body: bool = False):
        super().__init__()
        self.variance = variance
        self.outdim = outdim
        self.body = body
        self.backbone = slowfast101()
        state_dict = torch.load('models/SLOWFAST_R101_K700.pth.tar')
        keys = self.backbone.load_state_dict(state_dict, strict=False)
        print(keys)
        interim_dim = 2048
        
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm3d):
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False
        
        self.head = _Regression_head(interim_dim, self.outdim, self.variance)
        
    @autocast()
    def forward(self, x: torch.tensor, *args):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        slow, fast = self.backbone(x)
        feats = [slow, fast]
#         print(slow.shape, fast.shape)
        
        h, w = feats[0].shape[3:]
        feats = [nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, f.shape[1], h, w) for f in feats]
        feats = torch.cat(feats, dim=1)
#         print(feats.shape)
        
        out = self.head(feats)
        return out


class _Regression_head(nn.Module):
    def __init__(self, hidden_dim: int = 512, outdim: int = 1, variance: bool = True, kernel_size=3, mlp_1x1=False):
        super().__init__()
        self.variance = variance
        self.hidden_dim = hidden_dim
        self.att_norm = self.hidden_dim ** 0.5
        
        self.conv_reduce = nn.Conv2d(2304, self.hidden_dim, 1, bias=False)

        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size, padding=padding, bias=False)

        self.conv = nn.Conv2d(
            self.hidden_dim, self.hidden_dim,
            1 if mlp_1x1 else kernel_size,
            padding=0 if mlp_1x1 else padding,
            bias=False
        )
        self.norm = nn.GroupNorm(1, self.hidden_dim, affine=True)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dp = nn.Dropout(0.2)
        self.pred = nn.Linear(self.hidden_dim, outdim, bias=False)
    
    @autocast()
    def forward(self, x):
        # self-attention
        B = x.shape[0]
        x = self.conv_reduce(x)
        query = self.conv_q(x).unsqueeze(1)
        key = self.conv_k(x).unsqueeze(0)
        att = (query * key).sum(2) / self.att_norm
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(x)
        virt_feats = (att.unsqueeze(2) * value).sum(1)

        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)

        x = x + virt_feats
        
        x = self.pool(x)
        x = self.dp(x)
        x = x.view(B, -1)
#         print(x.shape)
        
        out = dict()
        
        pred = self.pred(x)
        out['pred'] = pred 
        return out
