import torch
import torch.nn as nn
import torchvision.models as models


class HybridResnet(nn.Module):
    def __init__(
        self, num_joints=21, img_feature_dim=512, kp_feature_dim=128, hidden_dim=512
    ):
        super(HybridResnet, self).__init__()

        # ResNet Backbone
        resnet = models.resnet18(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_fc = nn.Linear(512, img_feature_dim)

        # MLP FOR 2D FEATURES
        self.kp_mlp = nn.Sequential(
            nn.Linear(num_joints * 2, 256),
            nn.ReLU(),
            nn.Linear(256, kp_feature_dim),
            nn.ReLU(),
        )

        # FUSION BY CONCAT and MLP for 3D Coordinates
        fusion_dim = img_feature_dim + kp_feature_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints * 3),
        )

    def forward(self, images, keypoints_2d):

        # Features from images
        img_feat = self.cnn_backbone(images)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.cnn_fc(img_feat)

        # Features from keypoints
        kp_feat = keypoints_2d.view(keypoints_2d.size(0), -1)
        kp_feat = self.kp_mlp(kp_feat)

        # Fusion by concat
        fused = torch.cat([img_feat, kp_feat], dim=1)

        out = self.regressor(fused)
        out = out.view(-1, 21, 3)

        return out
