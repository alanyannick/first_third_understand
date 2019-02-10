import random
import math
import cv2
import numpy as np
import joblib
import torch
import torch.nn.functional as F
import pylab as pl
from utils import torch_utils

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def class_weights():  # frequency of each class in coco train2014
    weights = 1 / torch.FloatTensor(
        [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380])
    weights /= weights.sum()
    return weights


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):  # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def build_targets(
        pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # print('max')
            # print(torch.argmax(pred_conf,0))
            # max_ind= torch.argmax(pred_conf[b])
            # mask[b].view(-1)[max_ind]=1
            # conf_mask[b].view(-1)[max_ind]=1
            # Convert to position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # print(gt_box)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # print(pred_box)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    import torch
    a = torch.load(filename, map_location='cpu')
    a['optimizer'] = []
    torch.save(a, filename.replace('.pt', '_lite.pt'))


def coco_class_count(path='../coco/labels/train2014/'):
    # histogram of occurrences per class
    import glob

    nC = 80  # number classes
    x = np.zeros(nC, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nC)
        print(i, len(files))


# -------------- Visualize Part ----------------


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, channel=0):
    image_numpy = image_tensor[channel].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = cv2.cvtColor(np.transpose(image_numpy, (1, 2, 0)) * 255.0, cv2.COLOR_RGB2BGR)
    return image_numpy.astype(imtype)


def normalize_img(img_all):
    img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
    img_all = np.ascontiguousarray(img_all, dtype=np.float32)
    # img_all -= self.rgb_mean
    # img_all /= self.rgb_std
    img_all /= 255.0
    return img_all


def plot_results():
    # Plot YOLO training results file 'results.txt'
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    # import os; os.system('rm -rf results.txt && wget https://storage.googleapis.com/ultralytics/results_v1_0.txt')

    plt.figure(figsize=(16, 8))
    s = ['X', 'Y', 'Width', 'Height', 'Objectness', 'Classification', 'Total Loss', 'Precision', 'Recall', 'mAP']
    files = sorted(glob.glob('results*.txt'))
    for f in files:
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 7, 8, 17, 18, 16]).T  # column 16 is mAP
        n = results.shape[1]
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.plot(range(1, n), results[i, 1:], marker='.', label=f)
            plt.title(s[i])
            if i == 0:
                plt.legend()


def draw_bounding_box(image, bbox, thickness=1):
    WHITE = (255, 255, 255)
    bx, by, bw, bh = tuple(bbox)
    bx = int(bx)
    by = int(by)
    bw = int(bw)
    bh = int(bh)
    cv2.line(image, (bx, by), (bx + bw, by), WHITE, thickness)
    cv2.line(image, (bx, by), (bx, by + bh), WHITE, thickness)
    cv2.line(image, (bx, by + bh), (bx + bw, by + bh), WHITE, thickness)
    cv2.line(image, (bx + bw, by), (bx + bw, by + bh), WHITE, thickness)
    # cv2.imwrite('/home/yangmingwen/first_third_person/result.jpg', image)
    return image


def plot_one_box(x, img, color=None, label=None,
                 line_thickness=None):  # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    cv2.imwrite('/home/yangmingwen/first_third_person/result.jpg', img)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)


def print_current_predict(targets, model):
    gt_bbox = np.array(targets[0][0][1:5])
    print("Gt:")
    print(gt_bbox)
    img_size = 416
    gt_bbox *= img_size
    gt_bbox[0] = gt_bbox[0] - gt_bbox[2] / 2
    gt_bbox[1] = gt_bbox[1] - gt_bbox[3] / 2
    print("Gt:")
    print(gt_bbox)
    # get the cls label
    gt_label = np.array(targets[0][0][0])
    predict_label = model.classifier.pred_bbox[0]
    predict_bbox = model.classifier.pred_bbox[1]
    predict_bbox[0] = predict_bbox[0] - predict_bbox[2] / 2
    predict_bbox[1] = predict_bbox[1] - predict_bbox[3] / 2
    stride = 32
    predict_bbox[0] *= stride
    predict_bbox[1] *= stride
    predict_bbox[2] *= stride
    predict_bbox[3] *= stride
    print("predict_box:")
    print(predict_bbox)
    return gt_bbox, gt_label, predict_bbox, predict_label


def drawing_bbox_gt(input, bbox, label, name, vis):
    """
    input tensor(1,1,1,1), np.bbox(1,1,1,1)
    """
    # transfer from rgb 2 bgr, tensor 2 numpy
    scene_img_np = tensor2im(input)
    # Draw the box
    bbox_img = draw_bounding_box(scene_img_np, bbox)
    # Draw the bbox with keypoints
    bbox_img_with_keypoint = visual_keypoint(image=bbox_img, bbox=bbox, cluster_number=label)
    # transfer from bgr 2 rgb, numpy 2 tensor
    bbox_img = torch.from_numpy(bbox_img_with_keypoint).unsqueeze(0)
    # normalize image
    bbox_img = normalize_img(bbox_img)
    vis.image(bbox_img[0, :, :, :], win=name, opts=dict(title=name + ' images'))


def load_cluster_center_data(cluster_file):
    return joblib.load(cluster_file)


def visual_keypoint(image, bbox, cluster_number, cluster_file='./data/clusters'):
    # keypoints info
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 255 / 2, 255)
    PURPLE = (255, 0, 255)
    WHITE = (255, 255, 255)

    KEYPOINT_LABELS = [x.lower() for x in [
        "Head",  # 0
        "L-Shoulder",  # 1
        "R-Shoulder",  # 2
        "L-Elbow",  # 3
        "R-Elbow",  # 4
        "L-Wrist",  # 5
        "R-Wrist",  # 6
        "L-Hip",  # 7
        "R-Hip",  # 8
        "L-Knee",  # 9
        "R-knee",  # 10
        "L-Ankle",  # 11
        "R-Ankle",  # 12
        "TORSO"  # 13
        "ABDOM"  # 14
    ]]

    l_pair = [
        (0, 13),  # (0, 17),  # Head
        (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
        (13, 14),
        (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)
    ]

    line_color = [GREEN,
                  RED, RED, RED, RED, RED,
                  RED,
                  YELLOW, YELLOW, YELLOW, YELLOW, YELLOW]

    p_color = [GREEN,  # Head
               RED, RED, RED, RED, RED, RED,  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
               YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW,  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle,
               WHITE, WHITE]  # TORSO, ABDOM

    keypoints = access_keypoints(cluster_file, int(cluster_number))
    part_line = draw_keypoints(p_color, image, bbox, keypoints)
    keypoint_drawing_out = draw_limbs(image, part_line, l_pair, line_color)
    return keypoint_drawing_out


def access_keypoints(cluster_file, cluster_number):
    '''return keypoint data based on cluster_number, keypoints are relative to the center and size of the bbox'''
    results = []
    cluster_center_data = load_cluster_center_data(cluster_file)
    keypoints = cluster_center_data.cluster_centers_[cluster_number]
    # rescale to [-1, 1]
    keypoints /= np.max(np.abs(keypoints))
    # turn into pairs
    it = iter(keypoints)
    for x in it:
        results.append((x, next(it)))
    # create torso keypoint
    lshoulder = results[1]
    rshoulder = results[2]
    results.append(((lshoulder[0] + rshoulder[0]) / 2, (lshoulder[1] + rshoulder[1]) / 2))
    # create abdominal keypoint
    lhip = results[7]
    rhip = results[8]
    results.append(((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2))

    return results


def draw_keypoints(p_color, image, bbox, keypoints, thickness=2):
    bx, by, bw, bh = tuple(bbox)
    part_line = {}

    cx = bx + bw // 2
    cy = by + bh // 2

    for n in range(len(keypoints)):
        cor_x, cor_y = keypoints[n]
        part_line[n] = (int(cx + cor_x * (bw / 2)), int(cy + cor_y * (bh / 2)))
        # cv2.circle(image, (part_line[n][0], part_line[n][1]), 2, p_color[n], thickness)
    return part_line


def draw_limbs(image, part_line, l_pair, line_color, thickness=1):
    '''using the keypoints info on top'''
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(image, start_xy, end_xy, line_color[i], thickness)
    return image


def drawing_heat_map(input, prediction_all, vis, name, tmp_it=0):
    # curr_score_map = prediction_all[1][tmp_it].detach().cpu().numpy()
    # curr_cord_map = prediction_all[1][tmp_it].detach().cpu().numpy()
    box_pred = prediction_all[2][tmp_it].detach().cpu().numpy()
    box_corner = np.zeros(box_pred.shape)
    box_corner[:, :, :, 0] = box_pred[:, :, :, 0] - box_pred[:, :, :, 2] / 2
    box_corner[:, :, :, 1] = box_pred[:, :, :, 1] - box_pred[:, :, :, 3] / 2
    box_corner[:, :, :, 2] = box_pred[:, :, :, 0] + box_pred[:, :, :, 2] / 2
    box_corner[:, :, :, 3] = box_pred[:, :, :, 1] + box_pred[:, :, :, 3] / 2
    box_corner[box_corner < 0] = 0
    box_corner[box_corner > 13] = 13
    box_corner = box_corner.astype(int)
    my_sum = np.zeros((13, 13))
    my_max = np.zeros((13, 13))
    my_count = np.zeros((13, 13))
    for an_cnt in range(3):
        for x_cord in range(13):
            for y_cord in range(13):
                cur_box = box_corner[an_cnt, x_cord, y_cord, :]

                #        print(cur_box)
                my_sum[cur_box[1]:cur_box[3], cur_box[0]:cur_box[2]] = my_sum[cur_box[1]:cur_box[3],
                                                                       cur_box[0]:cur_box[2]] + \
                                                                       prediction_all[1][tmp_it][an_cnt][x_cord][
                                                                           y_cord].detach().cpu().numpy()
                my_max[cur_box[1]:cur_box[3], cur_box[0]:cur_box[2]] = np.maximum(
                    my_max[cur_box[1]:cur_box[3], cur_box[0]:cur_box[2]],
                    prediction_all[1][tmp_it][an_cnt][x_cord][y_cord].detach().cpu().numpy())
                my_count[cur_box[1]:cur_box[3], cur_box[0]:cur_box[2]] = my_count[cur_box[1]:cur_box[3],
                                                                         cur_box[0]:cur_box[2]] + 1
    my_heatmap = np.divide(my_sum, my_count)

    # height, width, _ = img.shape
    width = 416
    height = 416
    corresponding_map = cv2.resize(my_heatmap, (width, height))
    # normarlize the cmap
    corresponding_map -= corresponding_map.min()
    corresponding_map /= corresponding_map.max()
    # get the heatmap
    heatmap = cv2.applyColorMap(np.uint8(corresponding_map * 256)
                                , cv2.COLORMAP_JET)
    # transfer img dim
    scene_img_np = tensor2im(input)
    final_out = np.uint8(heatmap * 0.3 + scene_img_np * 0.5)
    heat_map = normalize_img(torch.from_numpy(final_out).unsqueeze(0))
    vis.image(heat_map[0, :, :, :], win=name, opts=dict(title=name + ' images'))
    return heatmap
