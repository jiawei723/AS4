import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO
        self.layers = nn.Sequential(
            F.Conv2d(1, 8, (3,3), stride=1),
            F.ReLU(),
            F.BatchNorm2d(3),
            F.ReLU(),
            F.Conv2d(8, 8, (3,3), stride=1),
            F.ReLU(),
            F.BatchNorm2d(3),
            F.ReLU(),
            F.Conv2d(8, 16, (5, 5), stride=2),
            F.ReLU(),
            F.BatchNorm2d(3),
            F.ReLU(),
            F.Conv2d(16, 16, (3, 3), stride=1),
            F.ReLU(),
            F.BatchNorm2d(3),
            F.ReLU(),
            F.Conv2d(16, 16, (3, 3), stride=1),
            F.ReLU(),
            F.BatchNorm2d(3),
            F.ReLU(),
            F.Conv2d(16, 32, (5, 5), stride=2),
            F.ReLU(),
            F.BatchNorm2d(3),
            F.ReLU(),
            F.Conv2d(32, 32, (3, 3), stride=1),
            F.ReLU(),
            F.BatchNorm2d(3),
            F.ReLU(),
            F.Conv2d(32, 32, (3, 3), stride=1),
            F.ReLU(),
            F.BatchNorm2d(3),
            F.ReLU(),
            F.Conv2d(32, 32, (3, 3), stride=1)

        )


    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        x = x.unsqueeze(1)
        x_pr = self.layers(x)
        return x_pr


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO
        self.layers0 = nn.Sequential(
            F.Conv2d(G, 8, (3,3), stride=1),
            F.ReLU()
        )
        self.layers1 = nn.Sequential(
            F.Conv2d(8, 16, (3, 3), stride=2),
            F.ReLU()
        )
        self.layers2 = nn.Sequential(
            F.Conv2d(16, 32, (3, 3), stride=2),
            F.ReLU()
        )
        self.layers3 = nn.Sequential(
            F.ConvTranspose2d(32, 16, (3, 3), stride=2),
            F.ReLU()
        )
        self.layers4 = nn.Sequential(
            F.ConvTranspose2d(16, 8, (3, 3), stride=2),
            F.ReLU()
        )
        self.layersfinal = nn.Sequential(
            F.Conv2d(8, 1, (3, 3), stride=1),
            F.ReLU()
        )


    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        s = x.unsqueeze(1)
        C0 = self.layers0(s)
        C1 = self.layers1(C0)
        C2 = self.layers2(C1)
        C3 = self.layers3(C2)
        C4 = self.layers4(C3+C1)
        s_bar = self.layersfinal(C4+C0)
        return s_bar

def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO
        x = x*rot + trans

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    warped_src_fea = F.grid_sample(src_fea, [x,y], mode='bilinear')

    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    B, C, H, W = ref_fea.size()
    B, C, D, H, W = warped_src_fea.size()
    ref_fea_s = np.split(ref_fea[B, :, H, W],G)
    warped_src_fea_s = np.split(warped_src_fea[B, :, D, H, W],G)
    S_g = G*np.dot(ref_fea_s, warped_src_fea_s)/C

    return S_g



def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    B, D, H, W = p.size()
    for j in range(D-1):
        reg_dep = reg_dep + depth_values[j]*p[j]

    return reg_dep

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    l1_loss = depth_est - depth_gt

    return l1_loss
