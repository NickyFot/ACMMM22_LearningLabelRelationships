import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.cuda.amp import autocast

from .ResNet import resnet50_scratch_dag
from .MixUp import MixUpLayer
from .PoseResNet import get_pose_net

def initi_params(layer):
    if hasattr(layer, 'weight'):
        nn.init.xavier_normal_(layer.weight, 0.001)
    if hasattr(layer, 'bias'):
        nn.init.constant_(layer.bias, 0.)

class Regressor(nn.Module):
    def __init__(self, outdim: int, variance: bool = True, body: bool = False, lfb: bool = False):
        super().__init__()
        self.variance = variance
        self.outdim = outdim
        self.body = body

        if body:
            self.backbone = resnet.resnet50()
            state_dict = torch.load('models/pose_resnet_50_256x256.pth.tar')
            keys = self.backbone.load_state_dict(state_dict, strict=False)
            print(keys)
            self.backbone = nn.Sequential(*(list(self.backbone.children())[:-1]))
            interim_dim = 2048
            self.conv1 = nn.Conv1d(in_channels=interim_dim, out_channels=interim_dim, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
        else:
            self.backbone = resnet50_scratch_dag()
            state_dict = torch.load('models/resnet50_vggface.pth')
            interim_dim = 2048
            new_dict = dict()
            for key in state_dict:
                new_dict[key.replace('module.', '')] = state_dict[key]
            keys = self.backbone.load_state_dict(new_dict, strict=False)
            print(keys)
            
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False
        
        #transformers
        self.head = _Regression_head(self.outdim, interim_dim, self.variance, lfb)

    @autocast()
    def forward(self, x: torch.tensor, **kwargs):
        B, T, C, H, W = x.shape
        x = x.contiguous()
        x = x.reshape(B*T, C, H, W)
        if not self.body:
            _, feats = self.backbone(x)
            feats = feats.reshape(B, T, -1)   
        else:
            feats = self.backbone(x)
            feats = feats.reshape(B, T, -1) 
            feats = feats.permute(0, 2, 1)
            feats = self.conv1(feats)
            feats = feats * (feats > 0)
            feats = feats.permute(0, 2, 1)
        
        out = self.head(feats, **kwargs)
        return out
    
    
class _Regression_head(nn.Module):
    def __init__(self, outdim: int, interim_dim: int, variance: bool = True, lfb: bool = False):
        super().__init__()
        self.variance = variance
        self.outdim = outdim
        self.interim = interim_dim
        self.lfb = lfb

        #transformers
        self.pos_emb = nn.Embedding(1000, interim_dim)
        self.emb_dropout = nn.Dropout(0.2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=interim_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)   
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        
        if self.lfb:
            self.nlb = _NLBlock(interim_dim)
            interim_dim *= 2
            self.fc_nlb = nn.Linear(in_features=interim_dim, out_features=interim_dim)
#         self.pred = nn.Linear(in_features=interim_dim, out_features=self.outdim)
        self.fc1 = nn.Linear(in_features=512*2, out_features=1)
        self.fc2 = nn.Linear(in_features=512*2, out_features=1)
        self.fc3 = nn.Linear(in_features=512*2, out_features=1)
        self.fc_total = nn.Linear(in_features=interim_dim, out_features=1)

        if self.variance:
            self.var = nn.Linear(in_features=interim_dim, out_features=self.outdim)
            nn.init.normal_(self.var.weight, mean=0, std=0.0001)
        
#         initi_params(self.pred)
    
    @autocast()
    def forward(self, x: torch.tensor, lfb_feats: torch.tensor = None, feature_extraction : bool = False):
        #transformer
        B, T, D = x.shape
        out = dict()
        embs = self.pos_emb.weight[:x.shape[1]]
        x += embs
        feats = self.emb_dropout(x)
        feats = self.transformer(feats)
        
        feats = feats.permute(0, 2, 1)
        feats = self.avg_pool(feats)
        feats = feats.reshape(B, -1)
        
        out['feats'] = feats
        
        if feature_extraction:
            return out
        
        if self.lfb:  
            feats = feats.unsqueeze(1)
            longterm = self.nlb(feats, lfb_feats)
            feats = torch.cat((feats, longterm), dim=2)
#             print(feats.shape)
            feats = feats.squeeze(1)
            feats = self.fc_nlb(feats)
        
        feats = self.dropout(feats)
#         pred = self.pred(feats)
#         out['pred'] = pred 
        feats_1 = feats[:, :1024]
        feats_2 = feats[:, 1024:2048]
        feats_3 = feats[:, 2048:3072]
        
        pred_1 = self.fc1(feats_1)
        pred_2 = self.fc2(feats_2)
        pred_3 = self.fc3(feats_3)
        pred_tot = self.fc_total(feats)
#         print(feats.shape, feats_1.shape)
        out['pred'] = torch.cat([pred_1, pred_2, pred_3, pred_tot], dim=1)
        if self.variance:
            var = self.var(feats)
            out['var'] = var
        return out
    

class _NLBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.infeats = in_features
        self.scale = 1/torch.sqrt(1/torch.tensor(self.infeats))
        self.theta = nn.Conv1d(in_channels=self.infeats, out_channels=self.infeats, kernel_size=1)
        self.phi = nn.Conv1d(in_channels=self.infeats, out_channels=self.infeats, kernel_size=1)
        self.gi = nn.Conv1d(in_channels=self.infeats, out_channels=self.infeats, kernel_size=1)
        self.ln = nn.LayerNorm(normalized_shape=self.infeats)
        self.fc = nn.Conv1d(in_channels=self.infeats, out_channels=self.infeats, kernel_size=1)
        self.drop = nn.Dropout(0.2)
    
    @autocast()
    def forward(self, x: torch.tensor, lfb_feats: torch.tensor):
        feats = x.permute(0, 2, 1)
        lfb_feats = lfb_feats.permute(0, 2, 1)
        
        theta = self.theta(feats) #N, D, T (16,  512, 1)
        phi = self.phi(lfb_feats) #N, D, T (16, 512, 3)
        gi = self.gi(lfb_feats)
        
        theta_phi = torch.einsum('bji,bjk->bik', theta, phi) #N, T, T (16, 1, 3)
        theta_phi *= self.scale
        theta_phi = F.softmax(theta_phi, dim=2)
        
        out = torch.einsum('bij,bkj->bik', theta_phi, gi) #N, C, D (16, 1, 512)
        out = self.ln(out)
        out = F.relu(out)
        
        out = out.permute(0, 2, 1)
        out = self.fc(out)
        out = self.drop(out)
        out += feats
        out = out.permute(0, 2, 1)
        return out
        
        