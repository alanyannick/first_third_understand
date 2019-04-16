# from models import Darknet
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
# import model
from torch.autograd import Variable
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg


class First_Third_Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # Build subnets whose input is RGB maps (images)
        model_name='/home/yangmingwen/first_third_person/first_third_understanding/retinanet_R-50-FPN_1x_model.pth'
        config_file ='/home/yangmingwen/first_third_person/first_third_understanding/retinanet_R-50-FPN_1x.yaml'
        cfg.merge_from_file(config_file)
        cfg.freeze()
        self.rgb = build_detection_model(cfg)
        self.rgb = torch.nn.DataParallel(build_detection_model(cfg))
        self.rgb.load_state_dict(torch.load(model_name)['model'])
        self.rgb = torch.nn.Sequential(*list(self.rgb.module.children())[:-1])
        self.rgb.train()
        self.rgb = self.rgb.cuda()

        # class branch
        self.classifier = ClassificationModel(256, num_classes=19)

        clean_weight = False
        if clean_weight:
            for subnet in (self.exo_sfn_conv, self.ego_ss_conv, self.exo_ss_conv, self.classifier):
                for param in subnet.parameters():
                    if param.dim() >= 4:
                        nn.init.xavier_uniform_(param)

    def forward(self, ego_rgb = None, exo_rgb = None, exo_rgb_gt = None, target = None, test_mode = False):
        self.test_mode = test_mode
        # GroundTruth
        self.targets = target
        self.exo_rgb_gt = exo_rgb_gt
        # change the list of tensor to 8x3x800x800
        self.ego_rgb = torch.stack(ego_rgb)[0]
        self.exo_rgb = torch.stack(exo_rgb)[0]
# ======================= get the feature pyramid here ==========
        with torch.no_grad():
            predictions = self.rgb(ego_rgb.unsqueeze(0).cuda())
# =======================First / Second  / third branch here =========================================
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



# class Retina_backbone(nn.Module):
#     def __init__(self):
#         nn.Module.__init__(self)
#         import pickle
#         from functools import partial
#         pickle.load = partial(pickle.load, encoding="latin1")
#         pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#         self.retinanet = torch.load('./model/coco_resnet_50_map_0_335.pt', map_location=lambda storage, loc: storage.cuda(0),
#                                pickle_module=pickle)
#         # self.rgb = nn.Sequential(*list(self.retinanet.children())[:-6])
#         self.features = []
#
#     def forward(self, input):
#         x = self.retinanet.conv1(input)
#         x = self.retinanet.bn1(x)
#         x = self.retinanet.relu(x)
#         x = self.retinanet.maxpool(x)
#
#         x1 = self.retinanet.layer1(x)
#         x2 = self.retinanet.layer2(x1)
#         x3 = self.retinanet.layer3(x2)
#         x4 = self.retinanet.layer4(x3)
#         self.features = self.retinanet.fpn([x2, x3, x4])
#         return self.features


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

if __name__ == '__main__':
    net = First_Third_Net()
    net(Variable(torch.randn(1, 3, 928, 640)).cuda().float())
    net()
