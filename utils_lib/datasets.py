import glob
import math
import os
import random
from sys import platform
import pickle
import cv2
import numpy as np
import torch
import random
import skimage.io
import skimage.transform
import skimage.color
import skimage
from skimage import img_as_float
from skimage import img_as_ubyte
import torchvision.transforms as transforms
# from torch.utils_lib.data import Dataset
from utils_lib.utils import xyxy2xywh
from torchvision import transforms as T
import torchvision.transforms as transforms


def process_index_minus(img_index_iter, img_index):
    img_index_iter = str(int(img_index_iter) - 1)
    # src_name = '00000.jpg' | input_name = '000'
    init_index = 5 - len(str(img_index_iter))
    final_index_name = list(img_index)
    start_index = 0
    for index in range(init_index, 5):
        final_index_name[index] = img_index_iter[start_index]
        start_index += 1
    final_replace_name = ''.join(str(e) for e in final_index_name)
    return final_replace_name

def process_index_plus(img_index_iter, img_index):
    img_index_iter = str(int(img_index_iter) + 1)
    # src_name = '00000.jpg' | input_name = '000'
    init_index = 5 - len(str(img_index_iter))
    final_index_name = list(img_index)
    start_index = 0
    for index in range(init_index, 5):
        final_index_name[index] = img_index_iter[start_index]
        start_index += 1
    final_replace_name = ''.join(str(e) for e in final_index_name)
    return final_replace_name

def normalize_img(img_all):
    img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
    img_all = np.ascontiguousarray(img_all, dtype=np.float32)
    img_all /= 255.0
    return img_all


def sv_augmentation(img, scene_img, scene_gt_img, fraction):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    scene_img_hsv = cv2.cvtColor(scene_img, cv2.COLOR_BGR2HSV)
    scene_gt_img_hsv = cv2.cvtColor(scene_gt_img, cv2.COLOR_BGR2HSV)

    S = img_hsv[:, :, 1].astype(np.float32)
    V = img_hsv[:, :, 2].astype(np.float32)

    S_scene = scene_img_hsv[:, :, 1].astype(np.float32)
    V_scene = scene_img_hsv[:, :, 2].astype(np.float32)

    S_scene_gt = scene_gt_img_hsv[:, :, 1].astype(np.float32)
    V_scene_gt = scene_gt_img_hsv[:, :, 2].astype(np.float32)

    a = (random.random() * 2 - 1) * fraction + 1

    S *= a
    S_scene *= a
    S_scene_gt *= a

    if a > 1:
        np.clip(S, a_min=0, a_max=255, out=S)
        np.clip(S_scene, a_min=0, a_max=255, out=S_scene)
        np.clip(S_scene_gt, a_min=0, a_max=255, out=S_scene_gt)

    a = (random.random() * 2 - 1) * fraction + 1

    V *= a
    V_scene *= a
    V_scene_gt *= a

    if a > 1:
        np.clip(V, a_min=0, a_max=255, out=V)
        np.clip(V_scene, a_min=0, a_max=255, out=V_scene)
        np.clip(V_scene_gt, a_min=0, a_max=255, out=V_scene_gt)

    img_hsv[:, :, 1] = S.astype(np.uint8)
    img_hsv[:, :, 2] = V.astype(np.uint8)

    scene_img_hsv[:, :, 1] = S_scene.astype(np.uint8)
    scene_img_hsv[:, :, 2] = V_scene.astype(np.uint8)

    scene_gt_img_hsv[:, :, 1] = S_scene_gt.astype(np.uint8)
    scene_gt_img_hsv[:, :, 2] = V_scene_gt.astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
    cv2.cvtColor(scene_img_hsv, cv2.COLOR_HSV2BGR, dst=scene_img)
    cv2.cvtColor(scene_gt_img_hsv, cv2.COLOR_HSV2BGR, dst=scene_gt_img)
    return img, scene_img, scene_gt_img


class load_images_and_labels():  # for training
    def __init__(self, path, batch_size=1, img_size=608, multi_scale=False, augment=False, shuffle_switch=True,
                 center_crop=False,
                 video_mask='/home/yangmingwen/first_third_person/nips_final_data/nips_data/per_video_gt_merged_train_nips.pickle',
                 frame_mask='/home/yangmingwen/first_third_person/nips_final_data/nips_data/final_nips_third_branch.pickle'):
        self.path = path
        # self.img_files = sorted(glob.glob('%s/*.*' % path))
        with open(path, 'r') as file:
            self.img_files = file.readlines()

        self.img_files = [path.replace('\n', '') for path in self.img_files]
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in
                            self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment
        self.shuffle_switch = shuffle_switch
        self.center_crop = center_crop
        assert self.nB > 0, 'No images found in path %s' % path

        # RGB normalization values
        # self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((1, 3, 1, 1))

        # Pre-processing transfomer
        TO_BGR255 = True
        if TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        # Values to be used for image normalization
        self.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
        self.PIXEL_STD = [1., 1., 1.]
        normalize_transform = T.Normalize(
            mean=self.PIXEL_MEAN, std=self.PIXEL_STD
        )

        if self.center_crop:
            self.transforms = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize((985, 985)),
                    # T.Resize((800, 800)),
                    T.ToTensor(),
                    to_bgr_transform,
                    normalize_transform,
                ]
            )

        else:
            self.transforms = T.Compose(
                [
                    T.ToPILImage(),
                    # T.Resize((985, 985)),
                    T.Resize((800, 800)),
                    T.ToTensor(),
                    to_bgr_transform,
                    normalize_transform,
                ]
            )

        # Pick mask
        affordance_flag = True
        if affordance_flag:
            # get the specific frame mask
            with open(frame_mask, 'rb') as gt_handle:
                gt_per_frame_mask = pickle.load(gt_handle)
                self.gt_per_frame_mask = gt_per_frame_mask

            with open(video_mask, 'rb') as gt_handle:
                gt_per_video_mask = pickle.load(gt_handle)
                self.gt_video_mask = gt_per_video_mask

            # with open(ignore_mask,
            #               'rb') as ignore_handle:
            #     ignore_per_video_mask = pickle.load(ignore_handle)
            #     self.ignore_video_mask = ignore_per_video_mask


    def __iter__(self):
        self.count = -1
        self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)

        if self.multi_scale:
            # Multi-Scale YOLO Training
            height = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
        else:
            # Fixed-Scale YOLO Training
            height = self.height

        img_all = []
        labels_all = []
        scene_all = []
        scene_gt_all = []
        frame_mask = []
        # ignore_mask = []
        video_mask = []

        for index, files_index in enumerate(range(ia, ib)):
            if self.shuffle_switch:
                img_path = self.img_files[self.shuffled_vector[files_index]]
                label_path = self.label_files[self.shuffled_vector[files_index]]
            else:
                img_path = self.img_files[files_index]
                label_path = self.label_files[files_index]

            scene_flag = True
            if scene_flag:
                scene_path = (img_path.split(img_path.split('-')[-1])[0] + '00001.jpg').replace('images', 'scenes').replace('first-', 'third-')
                scene_gt_path = (img_path).replace('images', 'scenes_gt').replace('first-', 'third-')
                scene_img = cv2.imread(scene_path)
                scene_gt_img = cv2.imread(scene_gt_path)
                if scene_img is None:
                    assert ("Cannot find the scene image in " + scene_path)
                    continue

            frame_flag = True
            # Get the specific frame mask here
            if frame_flag:
                try:
                    frame_tag = img_path.split('/')[-1].split('.jpg')[0]
                    per_frame_mask = self.gt_per_frame_mask[frame_tag]
                    per_frame_mask[per_frame_mask == -1] = 7

                except Exception as error:
                    print(error)
                    print('Dont care during training:' + frame_tag)
                    frame_tag = 'bwd-ZGWMEEGO_1_first-00560'
                    per_frame_mask = self.gt_per_frame_mask[frame_tag]
                    per_frame_mask[per_frame_mask == -1] = 7

            # Whiole image enter points here
            # Input "img == ego", "scene_img == ego", "self.per_video_mask == 13 x 13",  "self.per_video_ignore_mask == 13 x 13"
            pick_mask = True
            if pick_mask:
                video_tag = img_path.split(img_path.split('-')[-1])[0].split('/')[-1].split('first')[0]
                try:
                    self.per_video_mask = self.gt_video_mask[video_tag]
                    # self.per_video_ignore_mask = self.ignore_video_mask[video_tag]
                except:
                    print('Test_Mode: Do not care about the gt_video_mask'+video_tag)
                    video_tag = '0EU2AEGO_1_'
                    self.per_video_mask = self.gt_video_mask[video_tag]
                    # self.per_video_ignore_mask = self.ignore_video_mask[video_tag]

            # Input ego/exo/gt_exo
            # Ego 64 frames for I3D
            img = []
            img_name = img_path.split('/')[-1].split('first-')[-1]
            img_index = str(img_name.split('.jpg')[0])
            img_index_value = int(img_index)

            for index_value in range(0,-32,-1):
                # (0, -31)
                img_index_iter = str(img_index_value + index_value)
                # src_name = '00000.jpg' | input_name = '000'
                init_index = 5 - len(str(img_index_iter))
                final_index_name = list(img_index)
                start_index = 0
                for index in range(init_index, 5):
                    final_index_name[index] = img_index_iter[start_index]
                    start_index += 1
                final_replace_name = ''.join(str(e) for e in final_index_name)

                img_path_iter = img_path.replace(str(img_index), final_replace_name)
                img_i3d = cv2.imread(img_path_iter)

                if img_i3d is None:
                    print("cannot find the image" + img_path_iter)
                    # solve the -1 and > index video
                    final_replace_name = process_index_plus(img_index_iter, img_index)
                    previous_path_iter = img_path.replace(str(img_index), final_replace_name)

                    #@TODO Could be optimized with the "isDir"
                    prev_img = cv2.imread(previous_path_iter)

                    if prev_img is None:
                        # important Notice // Need to seriously gurantee here !!!
                        # using while should be really careful @Yangming
                        break_i = 0
                        while prev_img is None:
                            break_i += 1
                            final_replace_name = process_index_plus(img_index_iter, img_index)
                            previous_path_iter = img_path.replace(str(img_index), final_replace_name)
                            img_index_iter = int(img_index_iter) + 1

                            if int(img_index_iter) > int(img_index):
                                print('Cannot find source image, please check' + previous_path_iter)
                            prev_img = cv2.imread(previous_path_iter)

                            if break_i > 100:
                                break

                    img_i3d = prev_img
                else:
                    img_i3d = img_i3d

                img.append(img_i3d)
            # reverve the image
            img = img[::-1]

            img_plus = []
            for index_value in range(1, 33):
                # (1, 32)

                img_index_iter = str(img_index_value + index_value)
                # src_name = '00000.jpg' | input_name = '000'
                init_index = 5 - len(str(img_index_iter))
                final_index_name = list(img_index)
                start_index = 0
                for index in range(init_index, 5):
                    final_index_name[index] = img_index_iter[start_index]
                    start_index += 1
                final_replace_name = ''.join(str(e) for e in final_index_name)

                img_path_iter = img_path.replace(str(img_index), final_replace_name)
                img_i3d = cv2.imread(img_path_iter)

                if img_i3d is None:
                    print("cannot find the image" + img_path_iter)
                    # solve the -1 and > index video
                    final_replace_name  = process_index_minus(img_index_iter, img_index)
                    previous_path_iter = img_path.replace(str(img_index), final_replace_name)
                    prev_img = cv2.imread(previous_path_iter)

                    if prev_img is None:

                        break_i = 0
                        while prev_img is None:
                            break_i += 1
                            final_replace_name = process_index_minus(img_index_iter, img_index)
                            previous_path_iter = img_path.replace(str(img_index), final_replace_name)
                            img_index_iter = int(img_index_iter) - 1
                            if img_index_iter < int(img_index):
                                print('Cannot find source image, please check' + previous_path_iter)
                            prev_img = cv2.imread(previous_path_iter)

                            if break_i > 100:
                                assert "Something wrong happened"
                                break
                    img_i3d = prev_img

                else:
                    img_i3d = img_i3d
                img_plus.append(img_i3d)

            for i in range(0, len(img_plus)):
                img.append(img_plus[i])
            # Normalize the ego as I3D input
            if self.center_crop:
                for i in range(0, len(img)):
                    img[i] = cv2.resize(img[i], (256, 256)) / 128 - 1
            else:
                for i in range(0, len(img)):
                    img[i] = cv2.resize(img[i], (224, 224)) / 128 - 1

            scene_img = self.transforms(scene_img)
            scene_gt_img = self.transforms(scene_gt_img)

            # Achieve center crop here
            # center_crop = False
            if self.center_crop:
                x_crop1 = random.randint(0, 30)
                y_crop1 = random.randint(0, 30)
                x_crop2 = random.randint(0, 184)
                y_crop2 = random.randint(0, 184)
                x_crop3= round(float(x_crop2) * 16 / 985)
                y_crop3 = round(float(y_crop2) * 16 / 985)

                # Random crop for img // Note here will be W, H, C, need to be changed dims for I3D input
                for i in range(0, len(img)):
                    img[i] = img[i][y_crop1:y_crop1 + 224, x_crop1:x_crop1 + 224, :]

                scene_img = scene_img[:, y_crop2:y_crop2 + 800, x_crop2:x_crop2 + 800]
                scene_gt_img = scene_gt_img[:, y_crop2:y_crop2 + 800, x_crop2:x_crop2 + 800]

                # @TODO ix bug temporally for 13x13 input (should be 16*16 for video mask)
                self.per_video_mask = cv2.resize(self.per_video_mask.transpose(1,2,0), (16,16), interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
                self.per_video_mask = self.per_video_mask[:,y_crop3:y_crop3 + 13, x_crop3:x_crop3 + 13]

                # self.per_video_ignore_mask = self.per_video_ignore_mask[y_crop3:y_crop3 + 13, x_crop3:x_crop3 + 13]

                if frame_flag:
                    self.per_frame_mask = per_frame_mask[:, y_crop3:y_crop3 + 13, x_crop3:x_crop3 + 13]
            else:
                if frame_flag:
                    self.per_frame_mask = cv2.resize(per_frame_mask, (13, 13), interpolation=cv2.INTER_NEAREST)
                    # self.per_video_ignore_mask = cv2.resize(self.per_video_ignore_mask,(13, 13), interpolation=cv2.INTER_NEAREST)

            # Load labels and transfer it from CxCyWH to LxLyRxRy
            if os.path.isfile(label_path):
                labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
                labels1 = labels0.copy()
                labels = labels1.copy()
                w = 800
                h = 800
                ratio = 1.0
                labels[:, 1] = ratio * w * (labels1[:, 1] - labels1[:, 3] / 2)
                labels[:, 2] = ratio * h * (labels1[:, 2] - labels1[:, 4] / 2)
                labels[:, 3] = ratio * w * (labels1[:, 1] + labels1[:, 3] / 2)
                labels[:, 4] = ratio * h * (labels1[:, 2] + labels1[:, 4] / 2)
            else:
                labels = np.array([])

            nL = len(labels)
            if nL > 0:
                # convert xyxy to CxCywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy()) / height

            if self.augment:
                # random left-right flip
                lr_flip = False
                if lr_flip & (random.random() > 0.5):
                    img = np.fliplr(img)
                    scene_img = np.fliplr(scene_img)
                    if nL > 0:
                        labels[:, 1] = 1 - labels[:, 1]

                # random up-down flip
                ud_flip = False
                if ud_flip & (random.random() > 0.5):
                    img = np.flipud(img)
                    scene_img = np.flipud(scene_img)
                    if nL > 0:
                        labels[:, 2] = 1 - labels[:, 2]

            img_all.append(img)
            labels_all.append(labels)
            scene_all.append(scene_img)
            scene_gt_all.append(scene_gt_img)
            # ignore_mask.append(self.per_video_ignore_mask)
            video_mask.append(self.per_video_mask)
            frame_mask.append(self.per_frame_mask)

        # Transfer the label as contin
        labels_all = np.ascontiguousarray(labels_all, dtype=np.float32)
        try:
            # np.array img all will change it to 12 * 64 * 224 * 224 * 3 then transpose to 12 * 3 * 64 * 224 * 224
            return np.array(img_all).transpose(0, 4, 1, 2, 3), labels_all, scene_all, scene_gt_all, video_mask, frame_mask
            #         scene_all), torch.from_numpy(scene_gt_all)
        except:
            print('debug here something wrong happened / 411 line of datasets')


    def __len__(self):
        return self.nB  # number of batches


def resize_square(img, height=416, color=(0, 0, 0)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
    dw = height - new_shape[1]  # width padding
    dh = height - new_shape[0]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)  # resized, no border
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, dw // 2, dh // 2


def random_affine(img, scene_img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    sceneimgw = cv2.warpPerspective(scene_img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 1:5].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy, 0, height, out=xy)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return imw, sceneimgw, targets, M
    else:
        return imw




