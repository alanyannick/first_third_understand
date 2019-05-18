# from models import Darknet
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.nn.functional as F
# import model
from torch.autograd import Variable
import math
import sys
sys.path.append("/home/yangmingwen/first_third_person/first_third_understanding/networks/maskrcnn_benchmark/")

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg
import numpy as np

# I3D model for action recognition
import networks.i3dpt as i3dpt
from networks.i3dpt import I3D
from networks.i3dpt import Unit3Dpy


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
        # self.first_ego_pose_branch = egoFirstBranchModel(256, num_classes=3).cuda()

        # Build ego I3D module
        self.first_ego_pose_branch = egoFirstBranchModelI3D(num_classes=3).cuda()

        # Second Branch
        self.second_exo_affordance_branch = exoSecondBranchModel(256, num_classes=7).cuda()

        # Third Branch
        if self.third_branch_switch:
            self.third_affordance_branch = ThirdBranchModel(512, num_classes=8).cuda()

        # Merge Branch
        self.merge_feature = MergeFeatureModule(256).cuda()
        self.merge_feature_i3d = MergeFeatureI3d(256).cuda()
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
            weights = [1, 0.5, 0.25]
            class_weights = torch.FloatTensor(weights).cuda()
            self.ce_loss= nn.CrossEntropyLoss(weight=class_weights, size_average=True)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(size_average=True)
        self.soft_max = torch.nn.Softmax(dim=1)
        self.max_pooling = nn.MaxPool1d(7)
        self.sigmoid = nn.Sigmoid()

        # Important here with the 1x W x H x 8(3)
        self.mask_soft_max = torch.nn.LogSoftmax(dim=3)

        if with_weight_balance:
            # balance the weight from 0 - 1
            weights_mask = [1/0.0013080357142857143 / 1000,1/0.0013584449404761905 / 1000, 1/0.0011456473214285713 /1000,
                            1/0.0014114583333333334 /1000, 1/0.0017883184523809525 /1000, 1/0.0011772693452380952 /1000,
                            1/0.0011646205357142858/ 1000, 1/0.9906462053571429/1000]

            mask_weights = torch.FloatTensor(weights_mask).cuda()
            self.ce2d_loss= nn.CrossEntropyLoss(weight=mask_weights)
            self.nll_loss = nn.NLLLoss(weight=mask_weights)

        else:
            self.ce2d_loss = nn.CrossEntropyLoss(size_average=True)

        self.constrain_loss = ConstrainLoss()
        self.channel_contrain = False

        if self.channel_contrain:
            self.channel_constrain_loss = ConstrainLoss()

    def forward(self, ego_rgb = None, exo_rgb = None, exo_rgb_gt = None, target = None, video_mask = None, frame_mask = None, test_mode = False,
                mask_loss_switch = False, constain_switch=True):
        self.constrain_switch = constain_switch

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
        self.ego_rgb = torch.from_numpy(ego_rgb)
        self.exo_rgb = torch.stack(exo_rgb)

        # ====================== First Branch: ego pose
        # for cross_entropy / with out B X 1 X Class
        i3d_backbone_feature, ego_pose_out = self.first_ego_pose_branch(Variable(self.ego_rgb).cuda().float())
        retina_ego_features = self.merge_feature_i3d(i3d_backbone_feature).cuda()
        # Detach pose branch for avoiding influence
        detach_ego = False
        if detach_ego:
            retina_ego_features = retina_ego_features.detach()

        # Sigmoid the possiblity
        # Sigmoid = torch.nn.Sigmoid()
        # ego_pose_out = Sigmoid(ego_pose_out)

        # ====================== Second Branch: exo affordance

        # ======================= get the feature pyramid here ==========
        # with torch.no_grad():
        _, exo_feature_1, exo_feature_2, exo_feature_3, _ = self.rgb(self.exo_rgb.cuda())

        # Size 1 x 256 x 13 x13
        retina_exo_features = self.merge_feature(exo_feature_1, exo_feature_2, exo_feature_3).cuda()
        # get the mask_tensor here
        # ignore_mask = torch.from_numpy(np.array(ignore_mask)).float().cuda()
        frame_mask = torch.from_numpy(np.array(frame_mask)).squeeze(1).long().cuda()#type(torch.cuda.LongTensor)
        video_mask = torch.from_numpy(np.array(video_mask)).float().cuda()
        gt_video_mask = video_mask.permute(0, 2, 3, 1)
        # for binary_entropy / with out B X W X H X Class
        exo_affordance_out = self.second_exo_affordance_branch(retina_exo_features)
        # exo_affordance_out = torch.clamp(exo_affordance_out, 0, 1)

        # ====================== Third Branch: ego & exo affordance
        if self.third_branch_switch:
            pick_mask = False
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

                # After Sigmoid implemnt to get (0,1) weight * after softmax(0,1) pose weight
                channel_pick_mask = self.sigmoid(exo_affordance_out) * channel_pick_mask

            else:
                channel_pick_mask = 1
                # Get the possible region
                # channel_pick_mask = self.max_pooling(exo_affordance_out.contiguous().view(exo_affordance_out.shape[0], 13 * 13, 7)).view(
                #     exo_affordance_out.shape[0], 13, 13)
                # channel_pick_mask = channel_pick_mask.unsqueeze(3).repeat(1, 1, 1, 7).cuda()

            # Third branch with nB x Channel(256 X 2) X 13 X 13
            input_feature = torch.cat((retina_exo_features.cuda(), retina_ego_features.cuda()), dim=1).cuda()
            final_out_feature = self.third_affordance_branch(input_feature)

            # Final filter feature // clamp and log to avoid log(inf) // channel 7 will be the background channel
            # final_out_feature = self.soft_max(final_out_feature.cuda())
            final_out_feature = final_out_feature.cuda()

            #clamp
            # final_out_feature = torch.clamp(final_out_feature, 0, 1)

            # Create new tensor and put the value in to solve the inplace feature problem
            final_out_feature_final = torch.zeros(final_out_feature.shape).cuda()
            final_out_feature_final[:, :, :, :7] = torch.mul(final_out_feature[:, :, :, :7], channel_pick_mask)
            final_out_feature_final[:, :, :, 7] = final_out_feature[:, :, :, 7]
            # final_out_feature_final[:, :, :, 7] = torch.mul(final_out_feature[:, :, :, 7], (1 - channel_pick_mask.mean()))

            #clamp
            # final_out_feature_final = torch.clamp(final_out_feature_final, 0, 1)

            
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

            # Debuging
            mask_loss = self.ce2d_loss(final_out_feature_final.permute(0, 3, 1, 2), frame_mask).cuda()
            # print('min: value')
            # print(frame_mask.min())
            # print('max: value')
            # print(frame_mask.max())
            if mask_loss > 5:
                print('debug')
            # Constrain loss without background
            constain_loss = self.constrain_loss(final_out_feature_final[:,:,:,:7]).cuda()

            # Final loss
            if mask_loss_switch:
                if self.channel_contrain:
                    constain_loss_channel_mask = self.channel_constrain_loss(exo_affordance_out).cuda()
                    final_loss = pose_loss + affordance_loss + mask_loss + constain_loss + constain_loss_channel_mask
                    self.losses['constrain_loss'] = constain_loss + constain_loss_channel_mask
                else:
                    if self.constrain_switch:
                        final_loss = pose_loss + affordance_loss + mask_loss + constain_loss
                    else:
                        final_loss = pose_loss + affordance_loss + mask_loss

                    self.losses['constrain_loss'] = constain_loss

            else:
                # constain_affordance_loss =  self.channel_constrain_loss(exo_affordance_out).cuda()
                final_loss = pose_loss + affordance_loss # + constain_affordance_loss

            self.losses['pose_loss'] = pose_loss
            self.losses['affordance_loss'] = affordance_loss
            self.losses['mask_loss'] = mask_loss
            self.losses['constrain_loss'] = constain_loss

            return final_loss

        else:
            return torch.argmax(ego_pose_out, -1), self.sigmoid(exo_affordance_out), self.mask_soft_max(final_out_feature_final)


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

class egoFirstBranchModelI3D(nn.Module):

    def __init__(self, num_classes=3):
        super(egoFirstBranchModelI3D, self).__init__()
        # Build i3d model as ego backbone
        i3d_rgb_model_name='/home/yangmingwen/first_third_person/first_third_understanding/model/model_rgb.pth'
        i3d_rgb = I3D(num_classes=400, modality='rgb')
        # i3d_rgb.eval()
        i3d_rgb.load_state_dict(torch.load(i3d_rgb_model_name))
        # i3d_model backbone
        self.i3d_rgb = torch.nn.Sequential(*list(torch.nn.DataParallel(i3d_rgb).module.children())[:-4])

        # i3d_model final output branch submodule
        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))

        self.num_classes = num_classes
        self.dropout = torch.nn.Dropout(0)
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        self.i3d_softmax = torch.nn.Softmax(1)

    def forward(self, x):
        out_backbone = self.i3d_rgb(x)
        out = self.avg_pool(out_backbone)
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)
        out = out.squeeze(3)
        out = out.squeeze(3)
        out = out.mean(2)
        out_logits = out
        # out = self.softmax(out_logits)
        return out_backbone, out_logits


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
        # out = self.output_act(out)

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
        # out = self.output_act(out)

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
        # compress input should be B * C * H * W
        out7 = self.conv_compress(out)
        return out7

class MergeFeatureI3d(nn.Module):
    def __init__(self, feature_size=256):
        super(MergeFeatureI3d, self).__init__()
        # Compress channel
        self.conv_compress = nn.Conv2d(1024, feature_size, kernel_size=1)
        self.max_pooling = nn.MaxPool1d(8)
        self.upsampled = nn.Upsample(size=13, mode='bilinear')
    def forward(self, x):
        out = x.permute(0, 1, 3, 4, 2).contiguous().view(x.shape[0], 49 * 1024, 8)
        out = self.max_pooling(out).view(x.shape[0], 7, 7, 1024)
        # compress input should be B * C * H * W
        out = self.conv_compress(out.permute(0,3,1,2))
        # upscale 7x7 to 13x13
        out = self.upsampled(out)
        return out

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


class ConstrainLoss(nn.Module):
    def __init__(self):
        super(ConstrainLoss, self).__init__()
        self.grid_size = 14
        self.z = math.exp(math.log(2 * math.pi) + 1.)
        self.scaling = 1000
        self.loss = 0
        self.act = nn.Softmax(dim=3)
        self.protect_value = 0.000001

    def forward(self, feature_input):
        loss = 0
        feature_input = self.act(feature_input)
        # input B * W * H * C
        channel_size = feature_input.size()[3]
        batch_size = feature_input.size()[0]
        # create grid map 13 * 13
        xv, yv = torch.meshgrid([torch.arange(1, self.grid_size), torch.arange(1, self.grid_size)])
        # expand to feature channel grid, 13 * 13 * C
        xv = xv.unsqueeze(2).repeat(1, 1, channel_size).cuda()
        yv = yv.unsqueeze(2).repeat(1, 1, channel_size).cuda()

        # expand dim to batch
        xv = xv.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().cuda()
        yv = yv.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().cuda()

        for batch_index in range(0, batch_size):
            for channel_index in range(0, channel_size):
                # Calculate mass of x,y coordinate
                xv_energy_map = xv[batch_index,:,:,channel_index] * feature_input[batch_index,:,:,channel_index]
                mass_xv = xv_energy_map.sum() / (feature_input[batch_index,:,:,channel_index].sum() + self.protect_value)
                yv_energy_map = yv[batch_index,:,:,channel_index] * feature_input[batch_index,:,:,channel_index]
                mass_yv = yv_energy_map.sum() / (feature_input[batch_index,:,:,channel_index].sum() + self.protect_value)

                # Calculate covanrance
                # ((X - Xmean)^2 * k_weight).sum() / k_weight.sum()
                x_variance = (((xv[batch_index, :, :, channel_index] - mass_xv)).pow(2).float() * feature_input[batch_index,:,:,channel_index]).sum()
                # normalize
                x_variance = (x_variance / 169 / (feature_input[batch_index,:,:,channel_index].sum() + self.protect_value))

                # Calculate covanrance
                # # ((Y - Ymean)^2 * y_weight).sum() / k_weight.sum()
                y_variance = (((yv[batch_index, :, :, channel_index] - mass_yv)).pow(2).float() * feature_input[batch_index,:,:,channel_index]).sum()
                # normalize
                y_variance = (y_variance / 169 / (feature_input[batch_index,:,:,channel_index].sum() + self.protect_value))

                # Det xy == 2 * pi * e * (x + y) ^2 / (scaling_factor) * self.z (math.exp(math.log(2*math.pi) + 1.))
                det_xy = (x_variance + y_variance).pow(2) * self.z # .pow(2) # + self.z
                # Final loss
                loss += det_xy
        self.loss = loss / batch_size / channel_size
        return self.loss


if __name__ == '__main__':
    net = First_Third_Net()
    net(Variable(torch.randn(1, 3, 928, 640)).cuda().float())
    i3d_backbone_feature, output = net.first_ego_pose_branch(Variable(torch.randn(1, 3, 64, 224, 224)).cuda().float())
    net()
