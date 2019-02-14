import argparse

from models import *
from utils.datasets import *
from utils.utils import *
from utils.util import *
from networks.network import First_Third_Net
from utils import torch_utils
import visdom


def html_append_img(ims, txts, links, batch_i, out_img_folder, name='_exo.png', img=None):
    cv2.imwrite(os.path.join(out_img_folder, 'images_' + str(batch_i) + name), img)
    ims.append('images_' + str(batch_i) + name)
    txts.append('images_' + str(batch_i) + name + '<tr>')
    links.append('images_' + str(batch_i) + name)
    return ims, txts, links


def test(
        net_config_path,
        data_config_path,
        weights_file_path,
        out,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
        n_cpus=0,
        gpu_choice = "0",
        worker ='first',

):

    # Visualize Way
    # python -m visdom.server -p 8399
    vis = visdom.Visdom(port=8699)
    device = torch_utils.select_device(gpu_choice=gpu_choice)
    print("Using device: \"{}\"".format(device))

    # Make out path
    out_path = out
    mkdir(out_path)

    # Configure run
    data_config = parse_data_config(data_config_path)
    nC = int(data_config['classes'])  # number of classes (80 for COCO)
    test_path = data_config['valid']

    # Initiate model
    if worker == 'detection':
        model = Darknet(net_config_path, img_size)
    else:
        model = First_Third_Net(net_config_path)


    # Load weights
    if weights_file_path.endswith('.pt'):  # pytorch format
        checkpoint = torch.load(weights_file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        del checkpoint
    model.cuda().eval()

    # Get dataloader
    # dataset = load_images_with_labels(test_path)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpus)
    dataloader = load_images_and_labels(test_path, batch_size=batch_size, img_size=img_size, augment=True)

    mean_mAP, mean_R, mean_P = 0.0, 0.0, 0.0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class = [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    scene_flag = True
    ims = []
    txts = []
    links = []
    out_folder = os.path.join(out_path, 'web/')
    html = HTML(out_folder, 'final_out_html')
    html.add_header('First_Third_Person_Understanding')

    if scene_flag:
        for batch_i, (imgs, targets, scenes, scenes_gt) in enumerate(dataloader):

            with torch.no_grad():
                if worker == 'detection':
                    output = model(imgs.cuda())
                else:
                    output = model(imgs.cuda(),scenes.cuda(), scenes_gt, targets, test_mode=True)

                vis.image(model.exo_rgb[0, :, :, :], win="exo_rgb", opts=dict(title="scene_" + ' images'))
                vis.image(model.ego_rgb[0, :, :, :], win="ego_rgb", opts=dict(title="input_" + ' images'))
                vis.image(model.exo_rgb_gt[0, :, :, :], win="exo_rgb_gt", opts=dict(title="scene_gt_" + ' images'))
                gt_bbox, gt_label, predict_bbox, predict_label = print_current_predict(targets, model)

                gt_bbox_img_with_keypoint = drawing_bbox_gt(input=model.exo_rgb, bbox=gt_bbox, label=gt_label, name='gt_', vis=vis)
                predict_bbox_img_with_keypoint = drawing_bbox_gt(input=model.exo_rgb, bbox=predict_bbox, label=predict_label, name='predict_', vis=vis)
                heat_map = drawing_heat_map(input=model.exo_rgb, prediction_all=model.classifier.prediction_all, name='heat_map_',
                                 vis=vis)
                exo_rgb = tensor2im(model.exo_rgb)
                ego_rgb = tensor2im(model.ego_rgb)
                exo_rgb_gt = tensor2im(model.exo_rgb_gt)

                out_image_folder = os.path.join(out_folder,'images/')
                cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + '_exo.png'), exo_rgb)
                cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + '_ego.png'), ego_rgb)
                cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + '_exo_rgb_gt.png'), exo_rgb_gt)
                cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + 'gt_bbox_img_with_keypoint.png'), gt_bbox_img_with_keypoint)
                cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + '_predict_bbox_img_with_keypoint.png'), predict_bbox_img_with_keypoint)
                cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + '_heat_map.png'), heat_map)

                ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='_exo.png',
                                                   img=exo_rgb)
                ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='_ego.png',
                                                   img=ego_rgb)
                ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='_predict_bbox_img_with_keypoint.png',
                                                   img=predict_bbox_img_with_keypoint)
                ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='_heat_map.png',
                                                   img=heat_map)
                ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='_exo_rgb_gt.png',
                                                   img=exo_rgb_gt)
                ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='gt_bbox_img_with_keypoint.png',
                                                   img=gt_bbox_img_with_keypoint)


                html.add_images(ims, txts, links)
                html.save()
                ims = []
                txts = []
                links = []

    #             output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
    #
    #         # Compute average precision for each sample
    #         for sample_i, (labels, detections) in enumerate(zip(targets, output)):
    #             correct = []
    #
    #             if detections is None:
    #                 # If there are no detections but there are labels mask as zero AP
    #                 if labels.size(0) != 0:
    #                     mAPs.append(0), mR.append(0), mP.append(0)
    #                 continue
    #
    #             # Get detections sorted by decreasing confidence scores
    #             detections = detections.cpu().numpy()
    #             detections = detections[np.argsort(-detections[:, 4])]
    #
    #             # If no labels add number of detections as incorrect
    #             if labels.size(0) == 0:
    #                 # correct.extend([0 for _ in range(len(detections))])
    #                 mAPs.append(0), mR.append(0), mP.append(0)
    #                 continue
    #             else:
    #                 target_cls = labels[:, 0]
    #
    #                 # Extract target boxes as (x1, y1, x2, y2)
    #                 target_boxes = xywh2xyxy(labels[:, 1:5]) * img_size
    #
    #                 detected = []
    #                 for *pred_bbox, conf, obj_conf, obj_pred in detections:
    #
    #                     pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
    #                     # Compute iou with target boxes
    #                     iou = bbox_iou(pred_bbox, target_boxes)
    #                     # Extract index of largest overlap
    #                     best_i = np.argmax(iou)
    #                     # If overlap exceeds threshold and classification is correct mark as correct
    #                     if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
    #                         correct.append(1)
    #                         detected.append(best_i)
    #                     else:
    #                         correct.append(0)
    #
    #             # Compute Average Precision (AP) per class
    #             AP, AP_class, R, P = ap_per_class(tp=correct, conf=detections[:, 4], pred_cls=detections[:, 6],
    #                                               target_cls=target_cls)
    #
    #             # Accumulate AP per class
    #             AP_accum_count += np.bincount(AP_class, minlength=nC)
    #             AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)
    #
    #             # Compute mean AP across all classes in this image, and append to image list
    #             mAPs.append(AP.mean())
    #             mR.append(R.mean())
    #             mP.append(P.mean())
    #
    #             # Means of all images
    #             mean_mAP = np.mean(mAPs)
    #             mean_R = np.mean(mR)
    #             mean_P = np.mean(mP)
    #
    #             # Print image mAP and running mean mAP
    #             print(('%11s%11s' + '%11.3g' * 3) % (len(mAPs), dataloader.nF, mean_P, mean_R, mean_mAP))
    # else:
    #     for batch_i, (imgs, targets) in enumerate(dataloader):
    #         with torch.no_grad():
    #             output = model(imgs.cuda())
    #             output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
    #
    #         # Compute average precision for each sample
    #         for sample_i, (labels, detections) in enumerate(zip(targets, output)):
    #             correct = []
    #
    #             if detections is None:
    #                 # If there are no detections but there are labels mask as zero AP
    #                 if labels.size(0) != 0:
    #                     mAPs.append(0), mR.append(0), mP.append(0)
    #                 continue
    #
    #             # Get detections sorted by decreasing confidence scores
    #             detections = detections.cpu().numpy()
    #             detections = detections[np.argsort(-detections[:, 4])]
    #
    #             # If no labels add number of detections as incorrect
    #             if labels.size(0) == 0:
    #                 # correct.extend([0 for _ in range(len(detections))])
    #                 mAPs.append(0), mR.append(0), mP.append(0)
    #                 continue
    #             else:
    #                 target_cls = labels[:, 0]
    #
    #                 # Extract target boxes as (x1, y1, x2, y2)
    #                 target_boxes = xywh2xyxy(labels[:, 1:5]) * img_size
    #
    #                 detected = []
    #                 for *pred_bbox, conf, obj_conf, obj_pred in detections:
    #
    #                     pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
    #                     # Compute iou with target boxes
    #                     iou = bbox_iou(pred_bbox, target_boxes)
    #                     # Extract index of largest overlap
    #                     best_i = np.argmax(iou)
    #                     # If overlap exceeds threshold and classification is correct mark as correct
    #                     if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
    #                         correct.append(1)
    #                         detected.append(best_i)
    #                     else:
    #                         correct.append(0)
    #
    #             # Compute Average Precision (AP) per class
    #             AP, AP_class, R, P = ap_per_class(tp=correct, conf=detections[:, 4], pred_cls=detections[:, 6],
    #                                               target_cls=target_cls)
    #
    #             # Accumulate AP per class
    #             AP_accum_count += np.bincount(AP_class, minlength=nC)
    #             AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)
    #
    #             # Compute mean AP across all classes in this image, and append to image list
    #             mAPs.append(AP.mean())
    #             mR.append(R.mean())
    #             mP.append(P.mean())
    #
    #             # Means of all images
    #             mean_mAP = np.mean(mAPs)
    #             mean_R = np.mean(mR)
    #             mean_P = np.mean(mP)
    #
    #             # Print image mAP and running mean mAP
    #             print(('%11s%11s' + '%11.3g' * 3) % (len(mAPs), dataloader.nF, mean_P, mean_R, mean_mAP))
    #
    # # Print mAP per class
    # print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP') + '\n\nmAP Per Class:')
    #
    # classes = load_classes(data_config['names'])  # Extracts class labels from file
    # for i, c in enumerate(classes):
    #     print('%15s: %-.4f' % (c, AP_accum[i] / AP_accum_count[i]))
    #
    # # Return mAP
    # return mean_mAP, mean_R, mean_P


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')

    parser.add_argument('--data-config', type=str, default='cfg/person.data', help='path to data config file')
    parser.add_argument('--weights', type=str, default='weight_train_whole_data_2_13_batch_1/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.2, help='iou threshold for non-maximum suppression')
    parser.add_argument('--n-cpus', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--worker', type=str, default='first', help='size of each image dimension')
    parser.add_argument('--out', type=str, default='test_out_result/', help='cfg file path')
    parser.add_argument('--cfg', type=str, default='cfg/rgb-encoder.cfg,cfg/classifier.cfg', help='cfg file path')
    # parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='path to model config file')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    mAP = test(
        opt.cfg,
        opt.data_config,
        opt.weights,
        batch_size=opt.batch_size,
        img_size=opt.img_size,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        n_cpus=opt.n_cpus,
        out=opt.out,
    )
