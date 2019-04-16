# from models import Darknet
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import model
from torch.autograd import Variable
from maskrcnn-benchmark.maskrcnn_benchmark.modeling.detector import build_detection_model


class First_Third_Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # Build subnets whose input is RGB maps (images)
        # self.rgb = Darknet(dark_net_conf)
        model_name='./maskrcnn-benchmark/maskrcnn-benchmark/demo/retinanet_R-50-FPN_1x_model.pth'
	cfg='./maskrcnn-benchmark/configs/retinanet/retinanet_R-50-FPN_1x.yaml'
        self.rgb = torch.nn.DataParallel(build_detection_model(cfg))
        self.rgb.load_state_dict(torch.load(model_name)['model'])
	self.rgb=torch.nn.Sequential(*list(self.rgb.module.children())[:-1])
	self.rgb=self.rgb.cuda()
        #self.rgb = Retina_backbone().cuda()
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
        # self.classifier = Darknet(classifier_net_conf)
        for subnet in (self.exo_sfn_conv, self.ego_ss_conv, self.exo_ss_conv, self.classifier):
            for param in subnet.parameters():
                if param.dim() >= 4:
                    nn.init.xavier_uniform_(param)

        # Switch
        self.ss_feature_switch = False
        self.sfn_feature_switch = False

    def forward(self, ego_rgb = None, exo_rgb = None, exo_rgb_gt = None, target = None, test_mode = False):
        self.test_mode = test_mode

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
        if not test_mode:
            loss = self.classifier(concatted_features, self.targets, self.test_mode)
            return loss
        else:
            self.bbox_predict, [output, pred_conf, pred_boxes] = self.classifier(concatted_features, self.targets, self.test_mode)
            return self.bbox_predict, [output, pred_conf, pred_boxes]


class Retina_backbone(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        import pickle
        from functools import partial
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        self.retinanet = torch.load('../model/coco_resnet_50_map_0_335.pt', map_location=lambda storage, loc: storage.cuda(0),
                               pickle_module=pickle)
        # self.rgb = nn.Sequential(*list(self.retinanet.children())[:-6])
        self.features = []

    def forward(self, input):
        x = self.retinanet.conv1(input)
        x = self.retinanet.bn1(x)
        x = self.retinanet.relu(x)
        x = self.retinanet.maxpool(x)

        x1 = self.retinanet.layer1(x)
        x2 = self.retinanet.layer2(x1)
        x3 = self.retinanet.layer3(x2)
        x4 = self.retinanet.layer4(x3)
        self.features = self.retinanet.fpn([x2, x3, x4])
        return self.features


if __name__ == '__main__':
    # net = First_Third_Net()
    net = Retina_backbone().cuda()
    net(Variable(torch.randn(1, 3, 928, 640)).cuda().float())
    net()
