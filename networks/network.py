from models import Darknet
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from collections import defaultdict

class First_Third_Net(nn.Module):
    def __init__(self, net_conf):
        nn.Module.__init__(self)
        dark_net_conf = net_conf.split(',')[0]
        classifier_net_conf = net_conf.split(',')[1]

        # Build subnets whose input is RGB maps (images)
        self.rgb = Darknet(dark_net_conf)

        # Build subnets whose input is surface normal maps
        self.exo_sfn_conv = nn.Sequential(nn.Conv2d(3, 128, 9, stride=6), nn.Conv2d(128, 128, 5, stride=5))

        # Build subnets for ss_conv module
        self.ego_ss_conv = nn.Sequential(
            nn.Conv2d(2048, 256, 3, stride=2, padding=1),
            nn.Conv2d(256, 128, 3, stride=2, padding=1))

        self.exo_ss_conv = nn.Sequential(
            nn.Conv2d(2048, 256, 3, stride=2, padding=1),
            nn.Conv2d(256, 128, 3, stride=2, padding=1))

        # Build classifier subnet, which takes concatted features from earlier subnets
        self.classifier = Darknet(classifier_net_conf)
        for subnet in (self.exo_sfn_conv, self.ego_ss_conv, self.exo_ss_conv, self.classifier):
            for param in subnet.parameters():
                if param.dim() >= 4:
                    nn.init.xavier_uniform_(param)

        # Switch
        self.ss_feature_switch = False
        self.sfn_feature_switch = False

        # Data Parallel
        data_parallel = True
        if data_parallel:
            self.losses = {}
            self.rgb = nn.DataParallel(self.rgb)
            self.exo_sfn_conv = nn.DataParallel(self.exo_sfn_conv)
            self.ego_ss_conv = nn.DataParallel(self.ego_ss_conv)
            self.exo_ss_conv = nn.DataParallel(self.exo_ss_conv)
            self.classifier = nn.DataParallel(self.classifier)


    def forward(self, ego_rgb = None, exo_rgb = None, exo_rgb_gt = None, target = None):
        self.losses = defaultdict(float)
        self.exo_rgb = exo_rgb
        self.ego_rgb = ego_rgb
        self.targets = target
        self.exo_rgb_gt = exo_rgb_gt
        # Darknet feature
        ego_rgb = self.rgb(self.ego_rgb)
        exo_rgb = self.rgb(self.exo_rgb)

        # Original cat feature
        ego_cat = ego_rgb
        exo_cat = exo_rgb

        # Switch for adding ss & sfn feature
        if self.ss_feature_switch:
            # Semantic segmentation encoder
            ego_ss = self.semseg(self.ego_ss, return_feature_maps=False)
            ego_ss = self.ego_ss_conv(ego_ss[0])  # The segmentation encoder returns a list of one or more tensors
            exo_ss = self.semseg(self.exo_ss, return_feature_maps=False)
            exo_ss = self.exo_ss_conv(exo_ss[0])  # The segmentation encoder returns a list of one or more tensors
            # Concatenate features
            ego_cat = torch.cat([ego_rgb, ego_ss], 1)
            exo_cat = torch.cat([exo_rgb, exo_ss], 1)
        if self.sfn_feature_switch:
            # Surface normal
            exo_sfn = self.exo_sfn_conv(self.exo_sfn)
            exo_cat = torch.cat([exo_cat, exo_sfn], 1)

        concatted_features = torch.cat([ego_cat, exo_cat], 1)

        # @Verify the config file channel here
        # print(concatted_features.shape)
        # Note here, the targets dimension should be 1,1,5

        loss, self.pred_box, self.pred_all = self.classifier(concatted_features, self.targets)
        return sum(loss), self.pred_box, self.pred_all

if __name__ == '__main__':
    net = First_Third_Net()
    net()
