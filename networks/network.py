# from models import Darknet
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.nn.functional as F
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
        # ====================== detach the rgb gradient =========
        # self.rgb.detach()

        # First branch
        self.first_ego_pose_branch = egoFirstBranchModel(256, num_classes=19)

        # Second Branch
        self.second_exo_affordance_branch = exoSecondBranchModel(256, num_classes=19)

        # class branch
        self.classifier = ClassificationModel(256, num_classes=19)

        # Regression branch
        self.regressor = RegressionModel(256)

        # Compress the final channel
        # self.fc_ego_pool = nn.MaxPool2d(2,2)

        clean_weight = False
        if clean_weight:
            for subnet in (self.classifier, self.regressor):
                for param in subnet.parameters():
                    if param.dim() >= 4:
                        nn.init.xavier_uniform_(param)
        # Loss definition
        self.ce_loss= nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss(size_average=True)

    def forward(self, ego_rgb = None, exo_rgb = None, exo_rgb_gt = None, target = None, ignore_mask = None, video_mask = None, test_mode = False):
        self.test_mode = test_mode

        # GroundTruth
        self.targets = target

        # Get the Pose GT
        self.cls_targets = []
        for i in range(0, self.targets.shape[0]):
            # label should be long type :) @yangming
            self.cls_targets.append(self.targets[i][0][0])

        # Get the affordance GT

        self.exo_rgb_gt = exo_rgb_gt
        # change the list of tensor to 8x3x800x800
        self.ego_rgb = torch.stack(ego_rgb)
        self.exo_rgb = torch.stack(exo_rgb)
    # ======================= get the feature pyramid here ==========
        with torch.no_grad():
            retina_ego_features = self.rgb(self.ego_rgb.cuda())
            retina_exo_features = self.rgb(self.exo_rgb.cuda())
    # ======================@TBD design a feature merge module here to handle the multi-scale output of the retinaNet
        # merge_ego_feature_model = self.merge_feature(retina_ego_features)
        # merge_exo_feature_model = self.merge_feature(retina_exo_features)

    # ====================== First Branch: ego pose
        # for cross_entropy / with out B X 1 X Class
        ego_pose_out = self.first_ego_pose_branch(retina_ego_features[3])
        pose_loss = self.ce_loss(ego_pose_out, torch.LongTensor(self.cls_targets).cuda())

        import numpy as np
        if np.array(self.cls_targets).max() == 20:
            print('something wrong happened on target classes, which above the 19')
        # prediction = torch.argmax(first_ego_out, -1)
    # ====================== Second Branch: exo affordance
        # get the mask_tensor here
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).float().cuda()
        gt_ignore_mask = ignore_mask.repeat(19, 1, 1).view(8, 19, 13, 13).permute(0, 2, 3, 1)
        video_mask = torch.from_numpy(np.array(video_mask)).float().cuda()
        gt_video_mask = video_mask.permute(0, 2, 3, 1)

        # for binary_entropy / with out B X W X H X Class
        exo_affordance_out = self.second_exo_affordance_branch(retina_exo_features[3])
        affordance_loss = self.bce_loss(exo_affordance_out[gt_video_mask ==1], gt_video_mask[gt_video_mask == 1]) + \
                          self.bce_loss(exo_affordance_out[(gt_ignore_mask - gt_video_mask) == 1], gt_video_mask[(gt_ignore_mask - gt_video_mask) == 1])
        final_loss = pose_loss + affordance_loss
        self.losses = {}
        self.losses['pose_loss'] = pose_loss
        self.losses['affordance_loss'] = affordance_loss
        return final_loss

    # =======================First / Second  / third branch here =========================================
        # Switch for adding ss & sfn feature
        # concatted_features = torch.cat([retina_ego_features, retina_ego_features], 1)
        # ego_out = torch.cat([self.classifier(feature) for feature in retina_ego_features], dim=1)
        # ego_out = nn.AvgPool2d((ego_out.shape[-2:]))(ego_out)
        # F.interpolate(retina_exo_features, scale_factor=2, mode="nearest")
        # regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        # @Verify the config file channel here
        # print(concatted_features.shape)
        # Note here, the targets dimension should be 1,1,5
        # if not test_mode:
        #     loss = self.classifier(concatted_features, self.targets, self.test_mode)
        #     return loss
        # else:
        #     self.bbox_predict, [output, pred_conf, pred_boxes] = self.classifier(concatted_features, self.targets, self.test_mode)
        #     return self.bbox_predict, [output, pred_conf, pred_boxes]


class egoPoseClassification(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(egoPoseClassification, self).__init__()
        self.ego_pool = nn.AvgPool2d(13)
        self.fc = nn.Linear(255, 19)
        self.fc = nn.DataParallel(self.fc)
        F.interpolate(num_features_in, scale_factor=2, mode="nearest")


class egoFirstBranchModel(nn.Module):
    def __init__(self, num_features_in, num_classes=19, prior=0.01, feature_size=256):
        super(egoFirstBranchModel, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        # self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output = nn.Conv2d(feature_size, num_classes, kernel_size=3, padding=1)
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
        # Without the sigmoid
        # out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        # out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        # out2 = out1.view(batch_size, width, height, self.num_classes)
        # return out2.contiguous().view(x.shape[0], -1, self.num_classes)

        out2 = out1.view(batch_size, width, height, self.num_classes)
        # out is B x WHC
        out3 = out2.contiguous().view(x.shape[0], width * height * self.num_classes)
        # Out is 8 * 19
        out4 = nn.Linear(out3.shape[-1], self.num_classes).cuda()(out3)
        return out4


class exoSecondBranchModel(nn.Module):
    def __init__(self, num_features_in, num_classes=19, prior=0.01, feature_size=256):
        super(exoSecondBranchModel, self).__init__()

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        # self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output = nn.Conv2d(feature_size, num_classes, kernel_size=3, padding=1)
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
        # transfer out to B X W X H X C
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_classes)
        return out2


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


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

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

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)



if __name__ == '__main__':
    net = First_Third_Net()
    net(Variable(torch.randn(1, 3, 928, 640)).cuda().float())
    net()
