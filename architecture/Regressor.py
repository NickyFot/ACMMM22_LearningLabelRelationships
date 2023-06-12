"""pretrained """
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.cuda.amp import autocast

from .ResNet import resnet50_scratch_dag
from .ResNet import resnet50_face_sfew_dag


def initi_params(layer):
    if hasattr(layer, 'weight'):
        nn.init.xavier_normal_(layer.weight, 0.001)
    if hasattr(layer, 'bias'):
        nn.init.constant_(layer.bias, 0.)


class Backbone(nn.Module):
    def __init__(self, weights: str = 'models/resnet50_face_sfew_dag.pth', pretrained: str = "sfew", dim_red: bool = False):
        super().__init__()
        self.pretrained = pretrained
        self.reduce = dim_red
        if self.pretrained == 'sfew':
            self.backbone = resnet50_face_sfew_dag()
            self.interim_dim = 2048
        elif self.pretrained == 'vgg':
            self.backbone = resnet50_scratch_dag()
            self.interim_dim = 2048
        if self.reduce:
            self.interim_dim = 512
        state_dict = torch.load(weights)
        new_dict = dict()
        for key in state_dict:
            new_dict[key.replace('module.', '')] = state_dict[key]
        keys = self.backbone.load_state_dict(new_dict, strict=False)
        print(keys)
        self.dim_reduction = nn.Linear(in_features=2048, out_features=self.interim_dim)
        if self.reduce:
            self.bnorm1 = nn.BatchNorm1d(self.interim_dim)
            self.dout1 = nn.Dropout()
            self.dim_reduction2 = nn.Linear(in_features=self.interim_dim, out_features=self.interim_dim)
            self.bnorm2 = nn.BatchNorm1d(self.interim_dim)
            self.dout2 = nn.Dropout()
            # self.interim_dim = self.interim_dim // 2

        for module in self.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False

    @autocast()
    def forward(self, x: torch.tensor):
        B, T, C, H, W = x.shape
        x = x.contiguous()
        x = x.reshape(B * T, C, H, W)
        _, feats = self.backbone(x)
        feats = F.relu(self.dim_reduction(feats))
        if self.reduce:
            feats = self.bnorm1(feats)
            feats = self.dout1(feats)
            feats = F.relu(self.dim_reduction2(feats))
            feats = self.bnorm2(feats)
            feats = self.dout2(feats)
        feats = feats.reshape(B, T, -1)
        return feats


class Neck(nn.Module):
    def __init__(self, interim_dim):
        super().__init__()
        # transformers
        self.pos_emb = nn.Embedding(1000, interim_dim)
        self.emb_dropout = nn.Dropout(0.2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=interim_dim, nhead=16, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    @autocast()
    def forward(self, x: torch.tensor):
        # transformer
        B, T, D = x.shape
        embs = self.pos_emb.weight[:x.shape[1]]
        x_emb = x + embs
        feats = self.emb_dropout(x_emb)
        feats = self.transformer(feats)

        feats += x
        feats = feats.permute(0, 2, 1)
        feats = self.avg_pool(feats)
        feats = feats.reshape(B, -1)
        return feats


class NLB_Head(nn.Module):
    def __init__(self, interim_dim: int, variance: bool = True, scale: str = 'panss'):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(0.2)
        self.nlb = _NLBlock(interim_dim)
        interim_dim *= 2
        self.fc_nlb = nn.Linear(in_features=interim_dim, out_features=interim_dim)

        self.fc1 = nn.Linear(in_features=512 * 2, out_features=1)
        self.fc2 = nn.Linear(in_features=512 * 2, out_features=1)
        self.fc3 = nn.Linear(in_features=512 * 2, out_features=1)
        if self.scale == 'cains':
            self.fc4 = nn.Linear(in_features=512 * 2, out_features=1)  # cains only

        self.fc_total = nn.Linear(in_features=interim_dim, out_features=1)

        if variance:
            raise NotImplementedError
            # self.var = nn.Linear(in_features=interim_dim, out_features=self.outdim)
            # nn.init.normal_(self.var.weight, mean=0, std=0.0001)

    @autocast()
    def forward(self, x: torch.tensor, lfb_feats: torch.tensor = None):
        out = dict()
        feats = x.unsqueeze(1)
        longterm = self.nlb(feats, lfb_feats)
        feats = torch.cat((feats, longterm), dim=2)
        feats = feats.squeeze(1)
        feats = self.fc_nlb(feats)

        feats = self.dropout(F.relu(feats))
        feats_1 = feats[:, :512 * 2]
        feats_2 = feats[:, 512 * 2:512 * 4]
        feats_3 = feats[:, 512 * 4:512 * 6]
        feats_4 = feats[:, 512 * 6:]

        pred_1 = self.fc1(feats_1)
        pred_2 = self.fc2(feats_2)
        pred_3 = self.fc3(feats_3)
        if self.scale == 'cains':
            pred_4 = self.fc4(feats_4)  # cains
        pred_tot = self.fc_total(feats)
        if self.scale == 'panss':
            out['pred'] = torch.cat([pred_1, pred_2, pred_3, pred_tot], dim=1) # panss
        else:
            out['pred'] = torch.cat([pred_1, pred_2, pred_3, pred_4, pred_tot], dim=1)  # cains
        return out


class LN_Head(nn.Module):
    def __init__(self, interim_dim: int, variance: bool = True):
        super().__init__()
        self.variance = variance
        self.interim = interim_dim

        self.dropout = nn.Dropout(0.2)

        self.fc_nlb = nn.Linear(in_features=interim_dim, out_features=interim_dim)

        self.dis_size = interim_dim // 4
        self.fc1 = nn.Linear(in_features=self.dis_size, out_features=1)
        self.fc2 = nn.Linear(in_features=self.dis_size, out_features=1)
        self.fc3 = nn.Linear(in_features=self.dis_size, out_features=1)
        self.fc_total = nn.Linear(in_features=interim_dim, out_features=1)

        if self.variance:
            raise NotImplementedError
            # self.var = nn.Linear(in_features=interim_dim, out_features=self.outdim)
            # nn.init.normal_(self.var.weight, mean=0, std=0.0001)

    @autocast()
    def forward(self, x: torch.tensor, *args, **kwargs):
        out = dict()
        feats = self.fc_nlb(x)

        feats = self.dropout(F.relu(feats))
        feats_1 = feats[:, :self.dis_size]
        feats_2 = feats[:, self.dis_size:self.dis_size * 2]
        feats_3 = feats[:, self.dis_size * 2:self.dis_size * 3]

        pred_1 = self.fc1(feats_1)
        pred_2 = self.fc2(feats_2)
        pred_3 = self.fc3(feats_3)
        pred_tot = self.fc_total(feats)
        out['pred'] = torch.cat([pred_1, pred_2, pred_3, pred_tot], dim=1)
        return out


class LN_Head_VA(nn.Module):
    def __init__(self, interim_dim: int, variance: bool = True):
        super().__init__()
        self.variance = variance
        self.interim = interim_dim

        self.dropout = nn.Dropout(0.2)

        self.fc_nlb = nn.Linear(in_features=interim_dim, out_features=interim_dim)

        self.dis_size = interim_dim // 2
        self.fc1 = nn.Linear(in_features=self.dis_size, out_features=1)
        self.fc2 = nn.Linear(in_features=self.dis_size, out_features=1)


    @autocast()
    def forward(self, x: torch.tensor, *args, **kwargs):
        out = dict()
        feats = self.fc_nlb(x)

        feats = self.dropout(F.relu(feats))
        feats_1 = feats[:, :self.dis_size]
        feats_2 = feats[:, self.dis_size:]

        pred_1 = self.fc1(feats_1)
        pred_2 = self.fc2(feats_2)
        out['pred'] = torch.cat([pred_1, pred_2], dim=1)
        return out


class NLB_Head_AV(nn.Module):
    def __init__(self, interim_dim: int):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.nlb = _NLBlock(interim_dim)
        interim_dim *= 2
        self.fc_nlb = nn.Linear(in_features=interim_dim, out_features=interim_dim)

        self.dis_size = interim_dim // 2
        self.fc1 = nn.Linear(in_features=self.dis_size, out_features=1)
        self.fc2 = nn.Linear(in_features=self.dis_size, out_features=1)

    @autocast()
    def forward(self, x: torch.tensor, lfb_feats: torch.tensor = None):
        out = dict()
        feats = x.unsqueeze(1)
        longterm = self.nlb(feats, lfb_feats)
        feats = torch.cat((feats, longterm), dim=2)
        feats = feats.squeeze(1)
        feats = self.fc_nlb(F.relu(feats))

        feats = F.relu(feats)
        feats = self.dropout(feats)
        feats_1 = feats[:, :self.dis_size]
        feats_2 = feats[:, self.dis_size:]

        pred_1 = self.fc1(feats_1)
        pred_2 = self.fc2(feats_2)
        out['pred'] = torch.cat([pred_1, pred_2], dim=1)  # AV
        return out


class _NLBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.infeats = in_features
        self.scale = 1 / torch.sqrt(1 / torch.tensor(self.infeats))
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

        theta = self.theta(feats)  # N, D, T (16,  512, 1)
        phi = self.phi(lfb_feats)  # N, D, T (16, 512, 3)
        gi = self.gi(lfb_feats)

        theta_phi = torch.einsum('bji,bjk->bik', theta, phi)  # N, T, T (16, 1, 3)
        theta_phi *= self.scale
        theta_phi = F.softmax(theta_phi, dim=2)

        out = torch.einsum('bij,bkj->bik', theta_phi, gi)  # N, C, D (16, 1, 512)
        out = self.ln(out)
        out = F.relu(out)

        out = out.permute(0, 2, 1)
        out = self.fc(out)
        out = self.drop(out)
        out += feats
        out = out.permute(0, 2, 1)
        return out
