import argparse

from models import *
from utils_lib.datasets import *
from utils_lib.utils import *
from utils_lib.util import *
from networks.network import First_Third_Net
from utils_lib import torch_utils
import visdom
from scipy.io import loadmat
from networks import *
import networks
from sklearn.metrics import average_precision_score


def html_append_img(ims, txts, links, batch_i, i, out_img_folder, name='_exo.png', img=None):
    if img is not None:
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
        n_cpus=4,
        gpu_choice = "0",
        worker ='first',
        affordance_mode = False,
        shuffle_switch = False,
        testing_data = True,
        center_crop = False,


):
    # softMax function
    mask_soft_max = torch.nn.Softmax(dim=3)

    # Visualize Way
    # python -m visdom.server -p 8399
    vis = visdom.Visdom(port=8499)
    device = torch_utils.select_device(gpu_choice=gpu_choice)
    print("Using device: \"{}\"".format(device))

    # Make out path
    out_path = out
    mkdir(out_path)

    # Configure run
    data_config = parse_data_config(data_config_path)

    if testing_data == True:
        test_path = data_config['valid']
        pickle_video_mask = data_config['pickle_video_mask_test']
        # pickle_ignore_mask = data_config['pickle_ignore_mask_test']
        pickle_frame_mask = data_config['pickle_frame_mask_test']
    else:
        test_path = data_config['valid2']
        pickle_video_mask = data_config['pickle_video_mask_train']
        # pickle_ignore_mask = data_config['pickle_ignore_mask_train']
        pickle_frame_mask = '/home/yangmingwen/first_third_person/nips_final_data/nips_data/final_nips_third_branch.pickle'

    # Initiate model
    if worker == 'detection':
        model = Darknet(net_config_path, img_size)
    else:
        model = networks.network.First_Third_Net()


    # Load weights
    if weights_file_path.endswith('.pt'):  # pytorch format
        checkpoint = torch.load(weights_file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        del checkpoint
    model.cuda().eval()

    # Get dataloader
    dataloader = load_images_and_labels(test_path, batch_size=batch_size,
                                        img_size=img_size, augment=False, shuffle_switch=shuffle_switch,
                                        center_crop=center_crop,
                                        video_mask=pickle_video_mask,
                                        frame_mask=pickle_frame_mask
                                        )

    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    scene_flag = True
    ims = []
    txts = []
    links = []
    # create final folder
    out_folder = os.path.join(out_path, 'web8/')
    mkdir(out_folder)
    html = HTML(out_folder, 'final_out_html')
    html.add_header('First_Third_Person_Understanding')

    total_count = 0
    pose_correct_count = 0
    corrct_class_balance = []
    total_classes = []
    for class_it in range(0, 3):
        corrct_class_balance.append(0)
        total_classes.append(0)

    out_image_folder = '/home/yangmingwen/first_third_person/first_third_understanding/visual_map/result/'
    if scene_flag:
        calculate_map = True
        for batch_i, (imgs, targets, scenes, scenes_gt, video_mask, frame_mask) in enumerate(dataloader):
            if batch_i > 50:
                break
            else:
                try:
                    total_y_true = []
                    total_y_score = []
                    total_count += 1
                    if batch_i > 100000000:
                        break
                    with torch.no_grad():
                        if worker == 'detection':
                            output = model(imgs.cuda())
                        else:

                            pose_label, pose_affordance, frame_affordance= model(imgs, scenes, scenes_gt, targets, video_mask, frame_mask, test_mode=True)
                            predict_pose_label = pose_label.cpu().float().numpy()[0]
                            gt_pose_label = np.array(targets[0][0][0])

                            # Create the frame mask here
                            # Using frame mask to get the high_light_region
                            mask_different_class = []
                            mask_for_map = frame_mask[0][0]  # torch.argmax(frame_affordance, dim=3).cpu().float().numpy()[0]

                            # Assuming we only have one class & we want to get the final 1pose index of 0,1 mask ( out of 7 index mask)
                            for i in range(0, 7):
                                mask_different_class.append(((mask_for_map == i)).astype(int))

                                if ((mask_for_map == i)).astype(int).max() >= 1 :
                                    calculate_predict = i

                            # Get the pose label
                            predict_pose_label_3 = pose_label.cpu().float().numpy()[0]
                            gt_pose_label = np.array(targets[0][0][0])

                            # Transform targets from source to current 13 grid
                            # @@@@@ BUG here Notice!!!!! # @Date: 8.5
                            # x1 = (targets[0][0][1]/2 - targets[0][0][3] / 4) * 13
                            # x2 = (targets[0][0][1]/2 + targets[0][0][3] / 4) * 13
                            # y1 = (targets[0][0][2]/2 - targets[0][0][4] / 4) * 13
                            # y2 = (targets[0][0][2]/2 + targets[0][0][4] / 4) * 13
                            #
                            # if x1 < 0:
                            #     x1 = 0
                            # if y1 < 0:
                            #     y1 = 0
                            # if x2 > 13:
                            #     x2 = 13
                            # if y2 > 13:
                            #     y2 = 13
                            #
                            # gt_w = (x2 - x1)
                            # gt_h = (y2 - y1)
                            # gt_box_source = (x1/13 * 800, y1 /13* 800, gt_w/13*800, gt_h/13*800)


                             # Get the GT corresponding confident mask
                            frame_affordance_video = mask_soft_max(frame_affordance)
                            frame_heat_affordance = frame_affordance_video[:, :, :, :7].clone()

                            source_image = ((imgs[0][:, 32, :, :] + 1) * 128).transpose(1, 2, 0)
                            source_image = cv2.resize(source_image, (800,800))
                            # ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                            #                                    name='input_image_ego.jpg',
                            #                                    img=((imgs[0][:, 32, :, :] + 1) * 128).transpose(1, 2, 0))
                            # ----- Affordance prediction -------
                            visualize_affordance = affordance_mode
                            if visualize_affordance:

                                # gt_box visualization
                                # Find the hit or no-hit index
                                frame_mask = torch.from_numpy(np.array(frame_mask))
                                frame_mask[frame_mask == 7] = 0
                                width = 3 / 13 * 800
                                height = 5 / 13 * 800

                                # gt-box
                                index = frame_mask == frame_mask.max()
                                b, c, y_center_gt, x_center_gt = index.nonzero().tolist()[0]
                                c_gt = frame_mask.max().cpu().numpy()
                                # Calculate the coordinate [Left_Top X, Left_Top_y, w, h]
                                x_center_begin_gt = (x_center_gt - 1) / 13 * 800
                                y_center_begin_gt = (y_center_gt - 1) / 13 * 800
                                gt_box_gt = [int(x_center_begin_gt),int(y_center_begin_gt), int(width), int(height)]

                                # predict box
                                index = frame_heat_affordance == frame_heat_affordance.max()
                                b, y_center, x_center, c_predict= index.nonzero().tolist()[0]

                                # Calculate the coordinate [Left_Top X, Left_Top_y, w, h]
                                x_center_begin_predict = (x_center - 1) / 13 * 800
                                y_center_begin_predict = (y_center - 1) / 13 * 800

                                # hit or not
                                distance = np.square(x_center_gt - x_center) + np.square(y_center_gt-y_center)
                                if distance <= ((3*3+5*5)/2) :
                                    print('Hit Bounding Box')
                                    if c_gt == c_predict:
                                        predict_color = (0, 255, 0)
                                    else:
                                        predict_color = (255, 255, 0)
                                else:
                                    print('Not hit')
                                    predict_color = (0,0,255)

                                if calculate_map:
                                    # calculate MAP
                                    y_true = np.zeros((7, 13, 13), dtype=np.float16)
                                    for i in range(0, 13):
                                        for j in range(0, 13):
                                            if x_center_gt -3 < x_center < x_center_gt+3 and y_center_gt-3 < j < y_center_gt+3:
                                                print('Hit Check')
                                                # Set the region
                                                y_true[int(c_gt)][int(y_center -3):int(y_center+3),int(x_center -3):int(x_center+3)] = 1
                                    y_score = frame_affordance_video[:, :, :, :7].squeeze(
                                                0).permute(2, 0, 1).cpu().float().numpy()
                                    total_y_true.append(y_true)
                                    total_y_score.append(y_score)

                                # correct the pose
                                c_predict = c_predict
                                if c_predict == 4:
                                    pose_draw_label = 7
                                elif c_predict == 5:
                                    pose_draw_label = 12
                                elif c_predict == 6:
                                    pose_draw_label = 17
                                else:
                                    pose_draw_label = c_predict

                                # Draw pose with cut threshold
                                frame_heat_affordance[frame_heat_affordance < 0] = 0
                                sem_frame_affordance = cv2.resize(
                                    torch.max(frame_heat_affordance, dim=3)[0].cpu().float().numpy()[0], (800, 800))
                                sem_heatmap = cv2.applyColorMap(np.uint8(sem_frame_affordance * 255)
                                                                , cv2.COLORMAP_JET)
                                sem_heatmap_final_threshold = np.uint8(
                                    sem_heatmap * 0.3 + np.transpose((scenes[0] + 128).cpu().float().numpy(),
                                                                     (1, 2, 0)) * 0.6)

                                predict_bbox = [int(x_center_begin_predict), int(y_center_begin_predict), int(width),
                                                int(height)] # @notice xywh here

                                confidence = frame_heat_affordance.max()


                                predict_bbox_img_with_label_aff = drawing_bbox_keypoint_gt(input=list(torch.from_numpy(sem_heatmap_final_threshold - 128).permute(2, 0, 1).unsqueeze(0)),
                                                                                           bbox=predict_bbox,
                                                                                           label=pose_draw_label,color=predict_color, thickness=3)
                                predict_bbox_img_with_label_aff = cv2.putText(predict_bbox_img_with_label_aff,
                                                                              'Condidence: ' + str(
                                                                                  confidence.cpu().numpy()),
                                                                              (
                                                                              int(x_center_begin_predict), int(y_center_begin_predict)),
                                                                              cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                                                              predict_color,
                                                                              lineType=cv2.LINE_AA)

                                # ------------------------------ GT Box Visualiztion -----------------------
                                # gt_box visualization
                                # Find the maximum index
                                frame_mask = torch.from_numpy(np.array(frame_mask))
                                frame_mask[frame_mask == 7] = 0
                                index = frame_mask == frame_mask.max()
                                b, c, y_center, x_center = index.nonzero().tolist()[0]

                                # Calculate the coordinate [Left_Top X, Left_Top_y, w, h]
                                x_center_begin = (x_center - 1) / 13 * 800
                                y_center_begin = (y_center - 1) / 13 * 800
                                gt_box = [int(x_center_begin),int(y_center_begin), int(width), int(height)]
                                # using GT box
                                gt_label = frame_mask.max().cpu().numpy()

                                # correct the GT pose label
                                c = gt_label
                                if c == 4:
                                    gt_label = 7
                                elif c == 5:
                                    gt_label = 12
                                elif c == 6:
                                    gt_label = 17
                                else:
                                    gt_label = gt_label
                                gt_bbox_img_with_label_aff = drawing_bbox_keypoint_gt(input=list(
                                    (scenes_gt[0]).unsqueeze(0)),
                                                                                           bbox=gt_box,
                                                                                           label=gt_label,
                                                                                           color=(255,255,255))
                                gt_bbox_img_with_label_aff = cv2.putText(gt_bbox_img_with_label_aff,
                                                                              'GT',
                                                                              (
                                                                              int(x_center_begin), int(y_center_begin)),
                                                                              cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                                                              predict_color,
                                                                              lineType=cv2.LINE_AA)
                                final_image = np.concatenate((np.array(source_image, dtype=np.uint8), predict_bbox_img_with_label_aff),
                                               axis=1)
                                scenes_gt_np = np.transpose((scenes_gt[0] + 128).cpu().float().numpy(),
                                             (1, 2, 0))
                                final_image =  np.concatenate((final_image, gt_bbox_img_with_label_aff),
                                               axis=1)
                                # +'pose_'+str(c_predict)
                                cv2.imwrite(
                                    '/home/yangmingwen/first_third_person/first_third_understanding/visual_map_distance/'+'distance_reciprocal_'+str(1/distance)+'confidence_'
                                    +str(confidence.cpu().numpy())+'_'+str(batch_i)+
                                '.jpg', final_image)

                                out_image_folder = os.path.join(out_folder, 'images/')
                                ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                                   name='result_rank_with_scene.jpg',
                                                                   img=(final_image))

                            html.add_images(ims, txts, links)
                            html.save()
                            ims = []
                            txts = []
                            links = []

                except Exception as error:
                    print('something wrong happened' +error)

        if calculate_map:
            total_y_true = np.array(total_y_true)
            total_y_score = np.array(total_y_score)
            mAP = np.zeros((7), dtype=np.float16)
            for each_class in range(0, 7):
                per_y_score = total_y_score[:total_y_score.shape[0], each_class, :, :].reshape(
                    total_y_score.shape[0] * 13 * 13)
                per_y_true = total_y_true[:total_y_true.shape[0], each_class, :, :].reshape(
                    total_y_score.shape[0] * 13 * 13)
            # predict:
            print('Class' + str(each_class) + '_AP: ')
            # calculate MAP
            mAP[each_class] = average_precision_score(per_y_true, per_y_score)
            np.save('/home/yangmingwen/per_y_true_new' + str(each_class), per_y_true)
            np.save('/home/yangmingwen/per_y_true_new' + str(each_class), per_y_score)

            print('AP:')
            print(mAP)
            print('\nmAP:')
            print(mAP.mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')

    parser.add_argument('--data-config', type=str, default='cfg/person.data', help='path to data config file')
    parser.add_argument('--weights', type=str,
                        default='weight_retina_05_21_Pose_Affordance_Third_Final_Version_Loss_Switch_warm_up_2400/tmp_epo2_2500.pt',
                        help='path to weights file')
    parser.add_argument('--n-cpus', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--worker', type=str, default='first', help='size of each image dimension')
    parser.add_argument('--out', type=str, default='/home/yangmingwen/first_third_person/first_third_result/'
                                                   'weight_retina_08_06_Third_Final_Version_Loss_Switch_warm_up_2400/', help='cfg file path')
    parser.add_argument('--cfg', type=str, default='cfg/rgb-encoder.cfg,cfg/classifier.cfg', help='cfg file path')
    parser.add_argument('--testing_data_mode', type=bool, default=True, help='using testing or training data')
    parser.add_argument('--affordance_mode', type=bool, default=True, help='using testing or training data')
    parser.add_argument('--center_crop', type=bool, default=False, help='using testing or training data')
    # parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='path to model config file')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    mAP = test(
        opt.cfg,
        opt.data_config,
        opt.weights,
        affordance_mode=opt.affordance_mode,
        batch_size=opt.batch_size,
        img_size=opt.img_size,
        n_cpus=opt.n_cpus,
        out=opt.out,
        testing_data=opt.testing_data_mode,
        center_crop=opt.center_crop,
    )
