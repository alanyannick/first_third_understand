#############################
# Usage:
# Instantiate a model:
# 	net = Net()
# Then either load weights from a previous save:
# 	net.load(savedir)
# ...or load pre-trained weights into just the 3-chan subnets:
# 	net.load_pretrained_weights()
#############################

import sys, os
from cfg import *

from models import Darknet

# Import 3rd-party project CSAIL's semantic segmentation #
# sys.path.pop(0)
# sys.path.pop(0)
# print('semseg_models', semseg_models)
# sys.path.insert(0, cfg['semseg']['path'])
# sys.path.insert(0, os.path.join(cfg['semseg']['path'], 'csail_semseg'))
# from csail_semseg.models import models as semseg_models
# Ordinary imports (requiring no kludge on sys.path)
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

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

    def forward(self, ego_rgb = None, exo_rgb = None, target = None):

        self.exo_rgb = exo_rgb
        self.ego_rgb = ego_rgb
        self.targets = target

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
        loss = self.classifier(concatted_features, self.targets[0].unsqueeze(0))
        return loss

    # This function is for loading 3rd-party weights, not for weights that you have saved. It will only load weights into the rgb and sfn sub-networks.
    def load_pretrained_weights(self):
        fpath = "../weights/yolov3.weights"  # Use yolo_v3/weights/download_weights.sh to download these
        _3chan_darknets = [self.rgb]
        for subnet in _3chan_darknets:
            subnet.load_weights(
                fpath)



if __name__ == '__main__':
    net = First_Third_Net()
    net.load_pretrained_weights()
    net()
