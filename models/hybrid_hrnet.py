import torch
import torch.nn as nn
import timm


class HybridHRNet(nn.Module):
    def __init__(
        self, num_joints=21, img_feature_dim=512, kp_feature_dim=128, hidden_dim=512
    ):
        super(HybridHRNet, self).__init__()

        # HRNet Model
        self.cnn_backbone = timm.create_model(
            "hrnet_w32", pretrained=True, features_only=True
        )
        self.out_channels = self.cnn_backbone.feature_info[-1]["num_chs"]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.img_fc = nn.Linear(self.out_channels, img_feature_dim)

        # MLP for 2D keypoints
        self.kp_mlp = nn.Sequential(
            nn.Linear(num_joints * 2, 256),
            nn.ReLU(),
            nn.Linear(256, kp_feature_dim),
            nn.ReLU(),
        )

        # Fusion
        fusion_dim = img_feature_dim + kp_feature_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints * 3),
        )

    def forward(self, images, keypoints_2d):

        # HRNet Features
        feats = self.cnn_backbone(images)
        img_feat_map = feats[-1]
        img_feat = self.global_pool(img_feat_map)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.img_fc(img_feat)

        # 2D keypoints features
        kp_feat = keypoints_2d.view(keypoints_2d.size(0), -1)
        kp_feat = self.kp_mlp(kp_feat)

        # Fusion
        fused = torch.cat([img_feat, kp_feat], dim=1)

        out = self.regressor(fused)
        out = out.view(-1, 21, 3)

        return out
