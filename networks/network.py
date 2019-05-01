# from models import Darknet
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.nn.functional as F
# import model
from torch.autograd import Variable
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg
import numpy as np

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

        # Third branch switch
        self.third_branch_switch = True
        # First branch
        self.first_ego_pose_branch = egoFirstBranchModel(256, num_classes=3).cuda()

        # Second Branch
        self.second_exo_affordance_branch = exoSecondBranchModel(256, num_classes=7).cuda()

        # Third Branch
        if self.third_branch_switch:
            self.third_affordance_branch = ThirdBranchModel(512, num_classes=8).cuda()

        # Merge Branch
        self.merge_feature = MergeFeatureModule(256).cuda()
        # class branch
        # self.classifier = ClassificationModel(256, num_classes=19)

        # Regression branch
        # self.regressor = RegressionModel(256)

        # Compress the final channel
        # self.fc_ego_pool = nn.MaxPool2d(2,2)

        clean_weight = False
        if clean_weight:
            for subnet in (self.classifier, self.regressor):
                for param in subnet.parameters():
                    if param.dim() >= 4:
                        nn.init.xavier_uniform_(param)

        with_weight_balance = True
        if with_weight_balance:
        # Loss definition: Adds the pre-defined class weights
            weights = [0.375, 0.375, 1.0]
            class_weights = torch.FloatTensor(weights).cuda()
            self.ce_loss= nn.CrossEntropyLoss(weight=class_weights, size_average=True)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss(size_average=True)
        self.nll_loss = nn.NLLLoss(size_average=True)
        self.soft_max = torch.nn.Softmax()
        if with_weight_balance:
            weights_mask = [0.0009942434210526317, 0.0018601973684210526, 0.0009368832236842105, 0.0018332648026315789,
                        0.0018505345394736843, 0.0009594983552631579, 0.0009467516447368421,
                       0.9906186266447369]
            mask_weights = torch.FloatTensor(weights_mask).cuda()
            self.ce2d_loss= nn.CrossEntropyLoss(weight=mask_weights)

        else:
            self.ce2d_loss = nn.CrossEntropyLoss(size_average=True)
        self.log_soft_max = nn.LogSoftmax()

    def forward(self, ego_rgb = None, exo_rgb = None, exo_rgb_gt = None, target = None, ignore_mask = None, video_mask = None, frame_mask = None, test_mode = False,
                mask_loss_switch = False):
        self.test_mode = test_mode

        # GroundTruth
        self.targets = target

        # Get the Pose GT
        self.cls_targets = []
        for i in range(0, self.targets.shape[0]):
            # label should be long type :) @yangming
            self.cls_targets.append(self.targets[i][0][0])

        # Get the affordance GT
        self.losses = {}
        self.exo_rgb_gt = exo_rgb_gt
        # change the list of tensor to 8x3x800x800
        self.ego_rgb = torch.stack(ego_rgb)
        self.exo_rgb = torch.stack(exo_rgb)
        # ======================= get the feature pyramid here ==========
        # with torch.no_grad():
        _, ego_feature_1, ego_feature_2, ego_feature_3, _ = self.rgb(self.ego_rgb.cuda())
        _, exo_feature_1, exo_feature_2, exo_feature_3, _ = self.rgb(self.exo_rgb.cuda())

        # ======================@TBD design a feature merge module here to handle the multi-scale output of the retinaNet
        # Size 1 x 256 x 13 x13
        retina_ego_features = self.merge_feature(ego_feature_1, ego_feature_2, ego_feature_3).cuda()
        retina_exo_features = self.merge_feature(exo_feature_1, exo_feature_2, exo_feature_3).cuda()

        # ====================== First Branch: ego pose
        # for cross_entropy / with out B X 1 X Class
        ego_pose_out = self.first_ego_pose_branch(retina_ego_features)

        # Sigmoid the possiblity
        # Sigmoid = torch.nn.Sigmoid()
        # ego_pose_out = Sigmoid(ego_pose_out)

        # ====================== Second Branch: exo affordance
        # get the mask_tensor here
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).float().cuda()
        frame_mask = torch.from_numpy(np.array(frame_mask)).squeeze(1).long().cuda()#type(torch.cuda.LongTensor)
        video_mask = torch.from_numpy(np.array(video_mask)).float().cuda()
        gt_video_mask = video_mask.permute(0, 2, 3, 1)
        # for binary_entropy / with out B X W X H X Class
        exo_affordance_out = self.second_exo_affordance_branch(retina_exo_features)

        # ====================== Third Branch: ego & exo affordance
        if self.third_branch_switch:
            pick_mask = True
            if pick_mask:
                # Create channel_weight_mask firstly
                channel_pick_mask = torch.zeros(self.exo_rgb.shape[0], 13, 13, 7).cuda()
                ego_pick_mask = torch.zeros(self.exo_rgb.shape[0], 13, 13, 3).cuda()

                # Softmax to get the channel-wise wright
                ego_pose_out_softmax = self.soft_max(ego_pose_out).cuda()

                # Ego_pose0_weight with 0, 1, 2
                for batch_index in range(0, self.exo_rgb.shape[0]):
                    ego_pick_mask[batch_index, :, :, 0] = ego_pose_out_softmax[batch_index, 0].repeat(13, 13).cuda()
                    ego_pick_mask[batch_index, :, :, 1] = ego_pose_out_softmax[batch_index, 1].repeat(13, 13).cuda()
                    ego_pick_mask[batch_index, :, :, 2] = ego_pose_out_softmax[batch_index, 2].repeat(13, 13).cuda()

                    # copy the group weight from 0 1 2 to 0~7 dims
                    group_map = [0, 2, 5, 6]
                    for index in group_map:
                        channel_pick_mask[batch_index, :, :, index] = ego_pick_mask[batch_index, :, :, 0]
                    group_map = [1, 3]
                    for index in group_map:
                        channel_pick_mask[batch_index, :, :, index] = ego_pick_mask[batch_index, :, :, 1]
                    group_map = [4]
                    for index in group_map:
                        channel_pick_mask[batch_index, :, :, index] = ego_pick_mask[batch_index, :, :, 2]

                # Protect the gt label here
                if ego_pose_out.argmax() > 2:
                    assert ("Something wrong happened on GT datasets, maybe label out of index")

                # Get channel-wise pick mask
                channel_pick_mask = channel_pick_mask.cuda()
            else:
                channel_pick_mask = 1

            # Third branch with nB x Channel(256 X 2) X 13 X 13
            input_feature = torch.cat((retina_exo_features.cuda(), retina_ego_features.cuda()), dim=1).cuda()
            final_out_feature = self.third_affordance_branch(input_feature)

            # Final filter feature // clamp and log to avoid log(inf) // channel 7 will be the background channel
            # final_out_feature = self.soft_max(final_out_feature.cuda())
            final_out_feature = final_out_feature.cuda()

            # Create new tensor and put the value in to solve the inplace feature problem
            final_out_feature_final = torch.zeros(final_out_feature.shape).cuda()
            final_out_feature_final[:, :, :, :7] = torch.mul(final_out_feature[:, :, :, :7], channel_pick_mask)
            final_out_feature_final[:, :, :, 7] = final_out_feature[:, :, :, 7]

            
        if not test_mode:
            # Pose loss
            pose_loss = self.ce_loss(ego_pose_out, torch.LongTensor(self.cls_targets).cuda())

            # Affordance loss
            affordance_loss = self.bce_loss(exo_affordance_out[gt_video_mask == 1], gt_video_mask[gt_video_mask == 1]) + \
            self.bce_loss(exo_affordance_out[gt_video_mask == 0], gt_video_mask[gt_video_mask == 0])

            # Ignore mask loss
            # self.bce_loss(exo_affordance_out[(gt_ignore_mask - gt_video_mask) == 1], gt_video_mask[(gt_ignore_mask - gt_video_mask) == 1]).cuda()

            # Mask loss
            # Final feature
            # final_out_feature_filter = torch.log(torch.clamp(final_out_feature_final, min=0.00001, max=0.99999))
            mask_loss = self.ce2d_loss(final_out_feature_final.permute(0, 3, 1, 2), frame_mask).cuda()

            # Final loss
            if mask_loss_switch:
                final_loss = pose_loss + affordance_loss + mask_loss
            else:
                final_loss = pose_loss + affordance_loss
            self.losses['pose_loss'] = pose_loss
            self.losses['affordance_loss'] = affordance_loss
            self.losses['mask_loss'] = mask_loss
            return final_loss

        else:
            return torch.argmax(ego_pose_out, -1), exo_affordance_out, final_out_feature_final

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


class egoFirstBranchModel(nn.Module):
    def __init__(self, num_features_in, num_classes=4, prior=0.01, feature_size=256):
        super(egoFirstBranchModel, self).__init__()
        self.feature_size = feature_size
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
        self.output = nn.Conv2d(feature_size, int(feature_size/2), kernel_size=3, padding=1, stride=2)

        # Linear (# batch size)
        self.linear1 = nn.Linear(7*7*128, 4096)
        self.linear2 = nn.Linear(4096,  self.num_classes)

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
        batch_size, channels, width, height  = out.shape

        # out is B x WHC
        out1 = out.contiguous().view(x.shape[0], width * height * int(self.feature_size/2))
        # Out is 8 * 19
        out2 = self.linear1(out1)
        out3 = self.linear2(out2)
        return out3


class exoSecondBranchModel(nn.Module):
    def __init__(self, num_features_in, num_classes=15, prior=0.01, feature_size=256):
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


class ThirdBranchModel(nn.Module):
    def __init__(self, num_features_in, num_classes=7, prior=0.01, feature_size=512):
        super(ThirdBranchModel, self).__init__()

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


class MergeFeatureModule(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(MergeFeatureModule, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, stride=4, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()

        # Compress channel
        self.conv_compress = nn.Conv2d(768, feature_size, kernel_size=1)

    def forward(self, x1, x2, x3):
        out1 = self.conv1(x1)
        out2 = self.act1(out1)

        out3 = self.conv2(x2)
        out4 = self.act2(out3)

        out5 = self.conv3(x3)
        out6 = self.act3(out5)

        out = torch.cat((out2, out4, out6), dim=1)

        # B X W X H X C
        out7 = self.conv_compress(out)
        return out7


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
        # out = self.output_act(out)

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
