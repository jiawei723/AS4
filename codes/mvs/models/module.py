import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, (3,3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, (5, 5), stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (5, 5), stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)

        )


    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        x_pr = self.layers(x)
        return x_pr


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO
        self.layers0 = nn.Conv2d(G, 8, (3,3), stride=1, padding=1)
        self.layers1 = nn.Conv2d(8, 16, (3, 3), stride=2, padding=1)
        self.layers2 = nn.Conv2d(16, 32, (3, 3), stride=2, padding=1)
        self.layers3 = nn.ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1, output_padding=1)
        self.layers4 = nn.ConvTranspose2d(16, 8, (3, 3), stride=2, padding=1, output_padding=1)
        self.layersfinal = nn.Conv2d(8, 1, (3, 3), stride=1, padding=1)
        self.relu=nn.ReLU(True)

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        B, G, D, H, W = x.size()

        x = x.reshape(B,G,D*H,W)
        C0 = self.relu(self.layers0(x))
        C1 = self.relu(self.layers1(C0))
        C2 = self.relu(self.layers2(C1))
        C3 = self.layers3(C2)
        C4 = self.layers4(C3 + C1)
        s_bar = self.layersfinal(C4 + C0)
        outS = s_bar.reshape(B,D,H,W)
        return outS

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
        xyz = torch.stack((x, y, torch.ones_like(x)))
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)

        rot_xyz = torch.matmul(rot, xyz)
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, D, 1)
        depth = depth_values.unsqueeze(1).repeat(1,3,1).unsqueeze(3).repeat(1,1,1,H*W)
        rot_depth_xyz = rot_depth_xyz*depth
        proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)
        # avoid negative depth
        negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
        proj_xyz[:, 0:1][negative_depth_mask] = float(W)
        proj_xyz[:, 1:2][negative_depth_mask] = float(H)
        proj_xyz[:, 2:3][negative_depth_mask] = 1.0
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
        grid = proj_xy

        warped_src_fea = F.grid_sample(
        src_fea,
        grid.view(B, D * H, W, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return warped_src_fea.view(B, C, D, H, W)


def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    B, C, D, H, W = warped_src_fea.size()
    assert (C % G == 0)
    ref_fea_g = ref_fea.view(B, G, C // G, 1, H, W)
    warped_src_fea_g = warped_src_fea.view(B, G, C // G, D, H, W)
    similarity = (warped_src_fea_g * ref_fea_g).mean(2)

    return similarity



def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    return torch.sum(p * depth_values.view(depth_values.shape[0],depth_values.shape[1],1,1), dim=1)

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    loss = 0
    gt_masked = depth_gt[mask>0.5]
    est_masked = depth_est[mask>0.5]
    loss = F.smooth_l1_loss(est_masked, gt_masked)

    return loss

