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
from models import load_weights

class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # Build subnets whose input is 150-channel segmentation confidence maps
        # builder = semseg_models.ModelBuilder()
        # segmap_weights_encoder = os.path.join(cfg['semseg']['path'], opts.segmap_model_path,'encoder' + opts.segmap_suffix)
        # self.semseg = builder.build_encoder(
        #     arch=opts.segmap_arch_encoder,
        #     fc_dim=opts.segmap_fc_dim,
        #     weights=segmap_weights_encoder)
        #
        # # Freeze semantic segmentation module
        # for param in self.semseg.parameters():
        #     param.requires_grad = False

        # Build subnets whose input is RGB maps (images)
        self.rgb = Darknet('../cfg/rgb-encoder.cfg')

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
        self.classifier = Darknet('../cfg/classifier.cfg')
        for subnet in (self.exo_sfn_conv, self.ego_ss_conv, self.exo_ss_conv, self.classifier):
            for param in subnet.parameters():
                if param.dim() >= 4:
                    nn.init.xavier_uniform_(param)

        # Switch
        self.ss_feature_switch = False
        self.sfn_feature_switch = False
    # @arg targets must be supplied in training and omitted in testing
    # See yolo_v3/models.py and yolo_v3/utils/utils.py

    def set_input(self, batch_dict):
        self.targets = batch_dict.get('darknet_targets', None)
        self.ego_rgb = batch_dict['ego_rgb'].to(opts.device)
        self.exo_rgb = batch_dict['exo_rgb'].to(opts.device)
        self.exo_sfn = batch_dict['exo_sfnorm'].to(opts.device)
        self.ego_ss = batch_dict['ego_ss'].to(opts.device)
        self.exo_ss = batch_dict['exo_ss'].to(opts.device)

        # Visualiza the batch datasets here
        # ego_rgb_tensor = batch['ego_rgb']
        # ego_rgb_image = util.tensor2im(ego_rgb_tensor)
        # cv2.imwrite("/home/yangmingwen/first_third_person/ego.png", ego_rgb_image)

    def forward(self):
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
        loss = self.classifier(concatted_features, self.targets)

        return loss

    # This function is for loading 3rd-party weights, not for weights that you have saved. It will only load weights into the rgb and sfn sub-networks.
    def load_pretrained_weights(self):
        fpath = "../weights/yolov3.weights"  # Use yolo_v3/weights/download_weights.sh to download these
        _3chan_darknets = [self.rgb]
        for subnet in _3chan_darknets:
            subnet.load_weights(
                fpath)  # I checked the efficacy of this by ensuring a difference between the two printed lines from: (for p in subnet.parameters(): break); print(p.mean(), p.std()); subnet.load_weights(fpath); print(p.mean(), p.std())

    def load(self, savedir):
        self.backbone.load_weights(os.path.join(savedir, 'backbone'))
        self.rgb_1st.load_weights(os.path.join(savedir, 'rgb_1st'))
        self.rgb_3rd.load_weights(os.path.join(savedir, 'rgb_3rd'))
        self.sfn_1st.load_weights(os.path.join(savedir, 'sfn_1st'))
        self.sfn_3rd.load_weights(os.path.join(savedir, 'sfn_3rd'))
        self.segmap_1st.load_weights(os.path.join(savedir, 'segmap_1st'))
        self.segmap_3rd.load_weights(os.path.join(savedir, 'segmap_3rd'))


if __name__ == '__main__':
    net = Net()
    net.load_pretrained_weights()
    net()
