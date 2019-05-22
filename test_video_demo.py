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
import cv2
import numpy as np
import os

from os.path import isfile, join


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    files.sort(key=lambda x: int((x.split(".")[0])))

    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


# Giving input .txt will automatically generate the sequence images for each videos rather than shuffle here.

list_file = '/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/file_list_test_new_4_22.txt'
datasets_list = []
with open(list_file) as f:
    source_link = f.readlines()
    for index in range(0, len(source_link)):
        link = source_link[index]
        # get the ego_link'
        dataset_name = link.split('\t')[0].split('/')[-1].split('.jpg')[0].split('first')[0]
        if dataset_name in datasets_list:
            continue
        else:
            datasets_list.append(dataset_name)
print('datasets_list num:' + str(len(datasets_list)) + '\n' + str(datasets_list))

# =============================
# Video setting
out_file_link_balance = '/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/file_list_test_new_4_22_video_demo.txt'
set_video = 14
video_num = 1
init_frame = 400
number_frames = 1200
# Save Frame place
pathIn = '/home/yangmingwen/first_third_person/demo_video26/'
# Save Video place
pathOut = '/home/yangmingwen/first_third_person/demo/video_24.mp4'
# ==============================

Choose_Video = True
video_name = 'BD060003_'

with open(out_file_link_balance, 'w') as gt_file:
    with open(list_file) as f:
        source_link = f.readlines()
        for i in range(set_video, set_video+video_num):#len(datasets_list)):
            for index in range(init_frame, init_frame+number_frames):
                # '5UHV3EGO_5_first-00669.jpg

                img_index_iter = str(index)
                init_index = 5 - len(str(index))
                final_index_name = list('00000')
                start_index = 0
                for index in range(init_index, 5):
                    final_index_name[index] = img_index_iter[start_index]
                    start_index += 1
                final_replace_name = ''.join(str(e) for e in final_index_name)
                source_folder = '/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/images/'
                if Choose_Video:
                    link = source_folder + video_name + 'first-' + final_replace_name + '.jpg' + '\n'
                else:
                    link = source_folder + datasets_list[i] + 'first-' + final_replace_name + '.jpg' +'\n'
                gt_file.write(link)

gt_file.close()


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
        gpu_choice="0",
        worker='first',
        affordance_mode=False,
        shuffle_switch=False,
        testing_data=True,
        center_crop=False,

):
    # Visualize Way
    # python -m visdom.server -p 8399
    vis = visdom.Visdom(port=8499)
    device = torch_utils.select_device(gpu_choice=gpu_choice)
    print("Using device: \"{}\"".format(device))

    # softMax function
    mask_soft_max = torch.nn.Softmax(dim=3)
    # mask_hard_soft_max = torch.nn.Softmax(dim=3)

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

    test_path = out_file_link_balance

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

    if not os.path.exists(pathIn):
    #     os.remove(pathIn,*)
    #     os.removedirs(pathIn)
    #     os.mkdir(pathIn)
    # else:
        os.mkdir(pathIn)


    total_count = 0
    pose_correct_count = 0
    corrct_class_balance = []
    total_classes = []
    for class_it in range(0, 3):
        corrct_class_balance.append(0)
        total_classes.append(0)

    if scene_flag:
        for batch_i, (imgs, targets, scenes, scenes_gt, video_mask, frame_mask) in enumerate(dataloader):
            total_count += 1

            if batch_i > 10000000:
                break
            with torch.no_grad():
                if worker == 'detection':
                    output = model(imgs.cuda())
                else:
                    targets = np.array(list([          2,      1.0968,      0.7662,     0.32769,      1.2633]))[np.newaxis][np.newaxis]
                    # torch.save(scenes_gt,'/home/yangmingwen/first_third_person/scenes_gt.pth')

                    pose_label, pose_affordance, frame_affordance = model(imgs, scenes, scenes_gt, targets, video_mask,
                                                                          frame_mask, test_mode=True)
                    predict_pose_label = pose_label.cpu().float().numpy()[0]
                    gt_pose_label = np.array(targets[0][0][0])

                    # Bilinear version
                    # sem_frame_affordance_mask = torch.argmax(F.interpolate(frame_affordance.permute(0, 3, 1, 2), size=(800, 800), mode='bilinear').
                    #  permute(0, 2, 3, 1), dim=3).cpu().float().numpy()[0] + 24
                    # Masaic verison
                    sem_frame_affordance_mask = cv2.resize(
                        torch.argmax(frame_affordance, dim=3).cpu().float().numpy()[0], (800, 800),
                        interpolation=cv2.INTER_NEAREST) + 30

                    gt_sem_frame_affordance_mask = cv2.resize(frame_mask[0][0], (800, 800),
                                                              interpolation=cv2.INTER_NEAREST) + 30

                    colors = loadmat('data/color150.mat')['colors']
                    # colors[7] = np.array([0, 0, 0])
                    mask_switch = False
                    if mask_switch:
                        sem_frame_mask = colorEncode(sem_frame_affordance_mask, colors)
                        colors = loadmat('data/color150.mat')['colors']
                        gt_frame_mask = colorEncode(gt_sem_frame_affordance_mask, colors)

                    # get the accuracy
                    total_classes[int(gt_pose_label)] += 1
                    if predict_pose_label == gt_pose_label:
                        pose_correct_count += 1
                        corrct_class_balance[int(predict_pose_label)] += 1
                        print('Hit')

                    pose_affordance = pose_affordance.squeeze(0)
                    labelmap_rgb = np.zeros((800, 800),
                                            dtype=np.float16)
                    labelmap_rgb_gt = np.zeros((800, 800),
                                               dtype=np.float16)

                    each_map_threshold = 40

                    out_image_folder = os.path.join(out_folder, 'images/')


                    # Source image ego
                    # cv2.imwrite('/home/yangmingwen/first_third_person/1.jpg',
                    #             ((imgs[0][:, 32, :, :] + 1) * 128).transpose(1, 2, 0))

                    ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                       name='input_image_ego.jpg',
                                                       img=((imgs[0][:, 32, :, :] + 1) * 128).transpose(1, 2, 0))

                    ego_input = ((imgs[0][:, 32, :, :] + 1) * 128).transpose(1, 2, 0)

                    # group map
                    if gt_pose_label == 0:
                        group_map = [0, 2, 5, 6]
                    elif gt_pose_label == 1:
                        group_map = [1, 3]
                    else:
                        group_map = [4]

                    # Pose prediction
                    predict_bbox = [0, 0, 800, 800]
                    predict_label = predict_pose_label
                    if predict_label == 2:
                        predict_label += 5

                    if gt_pose_label == 2:
                        gt_label = gt_pose_label + 5
                    else:
                        gt_label = gt_pose_label

                    # predict_bbox_img_with_keypoint = drawing_bbox_keypoint_gt(input=scenes, bbox=predict_bbox,
                    #                                                           label=predict_label)
                    # gt_bbox_img_with_keypoint = drawing_bbox_keypoint_gt(input=scenes_gt, bbox=predict_bbox,
                    #                                                      label=gt_label)

                    # ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                    #                                    name='pose_prediction.jpg',
                    #                                    img=(predict_bbox_img_with_keypoint))
                    # ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                    #                                    name='pose_gt.jpg',
                    #                                    img=(gt_bbox_img_with_keypoint))

                    # Source image exo
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                       name='input_image'
                                                            + '_gt_label' + str(gt_pose_label) + '_predict_label' + str(
                                                           predict_pose_label) + '.jpg',
                                                       img=np.transpose((scenes_gt[0] + 128).cpu().float().numpy(),
                                                                        (1, 2, 0)))

                    input_exo_gt = np.transpose((scenes_gt[0] + 128).cpu().float().numpy(),
                                                                        (1, 2, 0))

                    # ----- Affordance prediction -------
                    visualize_affordance = affordance_mode
                    if visualize_affordance:
                        # heatmap prediction
                        heatmap_all = cv2.applyColorMap(np.uint8(labelmap_rgb)
                                                        , cv2.COLORMAP_JET)

                        # Mask prediction
                        if mask_switch:
                            ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                               name='predict_frame_mask_map.jpg',
                                                               img=(sem_frame_mask))

                        # frame_heat_affordance = frame_affordance.clone()
                        frame_affordance_video = mask_soft_max(frame_affordance)
                        frame_heat_affordance = frame_affordance_video[:, :, :, :7].clone()

                        # Set threhold for visualization
                        # frame_heat_affordance[frame_heat_affordance < 0] = 0
                        sem_frame_affordance = cv2.resize(
                            torch.max(frame_heat_affordance, dim=3)[0].cpu().float().numpy()[0], (800, 800))
                        sem_heatmap = cv2.applyColorMap(np.uint8(sem_frame_affordance * 255)
                                                        , cv2.COLORMAP_JET)
                        sem_heatmap_final = np.uint8(
                            sem_heatmap * 0.3 + np.transpose((scenes[0] + 128).cpu().float().numpy(), (1, 2, 0)) * 0.6)

                        # Find the maximum index
                        index = frame_heat_affordance == frame_heat_affordance.max()
                        b, x_center, y_center, c = index.nonzero().tolist()[0]

                        # Calculate the coordinate [Left_Top X, Left_Top_y, w, h]
                        x_center_begin = (x_center-1) / 13 * 800
                        y_center_begin = (y_center-1) / 13 * 800
                        x_center_end = 3 / 13 * 800
                        y_center_end = 2 / 13 * 800

                        # correct the pose
                        if c == 4:
                            pose_draw_label = 7
                        elif c == 5:
                            pose_draw_label = 12
                        elif c == 6:
                            pose_draw_label = 17
                        else:
                            pose_draw_label = c

                        # Draw pose with cut threshold
                        frame_heat_affordance[frame_heat_affordance < 0.4] = 0
                        sem_frame_affordance = cv2.resize(
                            torch.max(frame_heat_affordance, dim=3)[0].cpu().float().numpy()[0], (800, 800))
                        sem_heatmap = cv2.applyColorMap(np.uint8(sem_frame_affordance * 255)
                                                        , cv2.COLORMAP_JET)
                        sem_heatmap_final_threshold = np.uint8(
                            sem_heatmap * 0.3 + np.transpose((scenes[0] + 128).cpu().float().numpy(), (1, 2, 0)) * 0.6)


                        predict_bbox = [int(y_center_begin),int(x_center_begin), int(y_center_end), int(x_center_end)]
                        predict_bbox_img_with_label_aff= drawing_bbox_keypoint_gt(input=list(torch.from_numpy(sem_heatmap_final_threshold-128).permute(2, 0, 1).unsqueeze(0)), bbox=predict_bbox,
                                                                                  label=pose_draw_label)
                        # Put text
                        text = 'Pose' + str(c)
                        predict_bbox_img_with_label_aff = cv2.putText(predict_bbox_img_with_label_aff, text, (int(y_center_begin),int(x_center_begin)), cv2.FONT_HERSHEY_COMPLEX,
                                    1, (0, 255, 255), 1, cv2.LINE_AA)

                        text = 'Cut Confidence < 0.4'
                        predict_bbox_img_with_label_aff = cv2.putText(predict_bbox_img_with_label_aff, text, (5,5), cv2.FONT_HERSHEY_COMPLEX,
                                    1, (0, 255, 255), 1, cv2.LINE_AA)
                        ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                           name='predict_frame_mask_map_intensity.jpg',
                                                           img=(sem_heatmap_final))
                        # Mask gt
                        if mask_switch:
                            ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                               name='gt_frame_mask_map.jpg',
                                                               img=(gt_frame_mask))

                        ego_input = cv2.resize(ego_input, (800, 800))
                        final_total = np.concatenate([ego_input, sem_heatmap_final, predict_bbox_img_with_label_aff], axis=1)
                        # input_exo_gt, gt_frame_mask, sem_frame_mask

                        final_path = pathIn + str(batch_i) + '.png'
                        cv2.imwrite(final_path, final_total)

                    # html.add_images(ims, txts, links)
                    # html.save()
                    # ims = []
                    # txts = []
                    # links = []

        fps = 25.0
        convert_frames_to_video(pathIn, pathOut, fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')

    parser.add_argument('--data-config', type=str, default='cfg/person.data', help='path to data config file')
    parser.add_argument('--weights', type=str,
                        default='weight_retina_05_21_Pose_Affordance_Third_Final_Version_Loss_Switch_warm_up_2400/tmp_epo3_2500.pt',
                        help='path to weights file')
    # weight_retina_05_21_Pose_Affordance_Third_Final_Version_Loss_Switch_warm_up_2400/tmp_epo2_5000.pt
    parser.add_argument('--n-cpus', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--worker', type=str, default='first', help='size of each image dimension')
    parser.add_argument('--out', type=str,
                        default='/home/yangmingwen/first_third_person/first_third_result/weight_retina_05_21_Pose_Affordance_Third_Final_Version_Equal_Loss_Switch_warm_up_2400/',
                        help='cfg file path')
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
