import argparse

from models import *
from utils_lib.datasets import *
from utils_lib.utils import *
from utils_lib.util import *
from networks.network import First_Third_Net
from utils_lib import torch_utils
import visdom
from networks import *
import networks

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
        shuffle_switch = False,
        testing_data = True,

):

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
        pickle_ignore_mask = data_config['pickle_ignore_mask_test']
    else:
        test_path = data_config['train']
        pickle_video_mask = data_config['pickle_video_mask_train']
        pickle_ignore_mask = data_config['pickle_ignore_mask_train']

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
                                        video_mask=pickle_video_mask,
                                        ignore_mask=pickle_ignore_mask)

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
    if scene_flag:
        for batch_i, (imgs, targets, scenes, scenes_gt, ignore_mask, video_mask) in enumerate(dataloader):
            total_count += 1
            if batch_i > 500:
                break
            with torch.no_grad():
                if worker == 'detection':
                    output = model(imgs.cuda())
                else:
                    pose_label, pose_affordance= model(imgs, scenes, scenes_gt, targets, ignore_mask, video_mask, test_mode=True)
                    predict_pose_label = pose_label.cpu().float().numpy()[0]
                    gt_pose_label = np.array(targets[0][0][0])

                    if predict_pose_label == gt_pose_label:
                        pose_correct_count += 1
                        print('Hit')

                    pose_affordance = pose_affordance.squeeze(0)
                    labelmap_rgb = np.zeros((800, 800),
                                            dtype=np.float16)
                    labelmap_rgb_gt = np.zeros((800, 800),
                                            dtype=np.float16)

                    each_map_threshold = 40

                    out_image_folder = os.path.join(out_folder, 'images/')

                    # for i in range(0, pose_affordance.shape[2]):
                    for i in range(0, 7):
                        # predict for affordance
                        affordance = cv2.resize((pose_affordance[:, :, i].cpu().float().numpy() * 255), (800,800))
                        # ground truth for affordance
                        video_mask_gt = cv2.resize((video_mask[0][i,:,:] * 255.0), (800,800))

                        # insert pose_prediction and gt to html
                        ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                           name='pose_predict'+str(i)+'.jpg',
                                                           img=affordance)
                        ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                           name='pose_' + str(i) + '_gt.jpg',
                                                           img=video_mask_gt)
                        # insert the prediction_all
                        heatmap = cv2.applyColorMap(np.uint8(affordance)
                                                    , cv2.COLORMAP_JET)
                        ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                           name='predict_pose_heat_map'+str(i)+'.jpg',
                                                           img=heatmap)

                        # generate final affordance mask
                        labelmap_rgb[affordance >= each_map_threshold] = affordance[affordance >= each_map_threshold]

                        # generate final gt mask
                        labelmap_rgb_gt[video_mask_gt >= 0] = affordance[video_mask_gt >= 0]


                    # Source image ego & exo
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                       name='input_image_ego.jpg',
                                                       img=np.transpose((imgs[0]+128).cpu().float().numpy(), (1,2,0)))
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                       name='input_image'
                                    + '_gt_label' + str(gt_pose_label) + '_predict_label' + str(predict_pose_label) +'.jpg',
                                                       img=np.transpose((scenes_gt[0]+128).cpu().float().numpy(), (1,2,0)))


                    # heatmap prediction
                    heatmap_all = cv2.applyColorMap(np.uint8(labelmap_rgb)
                                                    , cv2.COLORMAP_JET)

                    # predict affordance
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                       name='predict_affordance_heat_map.jpg', img=heatmap_all)

                    # Get the gt_affordance
                    labelmap_rgb_gt = cv2.applyColorMap(np.uint8(labelmap_rgb_gt)
                                                    , cv2.COLORMAP_JET)
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                       name='gt_affordance_heat_map.jpg', img=labelmap_rgb_gt)

                    # Pick the channel prediction
                    affordance = cv2.resize((pose_affordance[:, :, pose_label].cpu().float().numpy() * 255), (800, 800))
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                       name='pick_label_prediction.jpg', img=affordance)

                    # Pick the channel groundtruth
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, i, out_image_folder,
                                                       name='pose_gt_' + str(int(gt_pose_label)) + '.jpg')
                    html.add_images(ims, txts, links)
                    html.save()
                    ims = []
                    txts = []
                    links = []

                # cv2.imwrite('/home/yangmingwen/first_third_person/pose' + str(300) + '.jpg',
                # cv2.cvtColor(affordance, cv2.COLOR_GRAY2RGB)* scenes_tmp + scenes_tmp)
                # @TBD ===========
                # get the final affordance
                debug_region = True
                if not debug_region:
                    final_affordace = torch.argmax(pose_affordance, 2).cpu().float().numpy()
                    cv2.resize((final_affordace), (800, 800))
                    heatmap = cv2.applyColorMap(np.uint8(affordance)
                                                , cv2.COLORMAP_JET)
                    # transfer img dim
                    scene_img_np = tensor2im(input)
                    final_out = np.uint8(heatmap * 0.3 + scene_img_np * 0.5)
                    vis.image(model.exo_rgb[0, :, :, :], win="exo_rgb", opts=dict(title="scene_" + ' images'))
                    vis.image(model.ego_rgb[0, :, :, :], win="ego_rgb", opts=dict(title="input_" + ' images'))
                    vis.image(model.exo_rgb_gt[0, :, :, :], win="exo_rgb_gt", opts=dict(title="scene_gt_" + ' images'))
                    gt_bbox, gt_label, predict_bbox, predict_label = print_current_predict(targets, model)

                    gt_bbox_img_with_keypoint = drawing_bbox_gt(input=model.exo_rgb, bbox=gt_bbox, label=gt_label, name='gt_', vis=vis)
                    predict_bbox_img_with_keypoint = drawing_bbox_gt(input=model.exo_rgb, bbox=predict_bbox, label=predict_label, name='predict_', vis=vis)
                    # heat_map = drawing_heat_map(input=model.exo_rgb, prediction_all=model.classifier.prediction_all, name='heat_map_',
                    #                    vis=vis)
                    exo_rgb = tensor2im(model.exo_rgb)
                    ego_rgb = tensor2im(model.ego_rgb)
                    exo_rgb_gt = tensor2im(model.exo_rgb_gt)

                    out_image_folder = os.path.join(out_folder,'images/')
                    cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + '_ego.png'), ego_rgb)
                    cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + '_exo.png'), exo_rgb)
                    # cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + '_heat_map.png'), heat_map)
                    cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + '_exo_rgb_gt.png'), exo_rgb_gt)
                    cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + 'gt_bbox_img_with_keypoint.png'), gt_bbox_img_with_keypoint)
                    cv2.imwrite(os.path.join(out_image_folder, 'images_' + str(batch_i) + '_predict_bbox_img_with_keypoint.png'), predict_bbox_img_with_keypoint)

                    ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='_ego.png',
                                                       img=ego_rgb)
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='_exo.png',
                                                       img=exo_rgb)
                    # ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='_heat_map.png',
                    #                                          img=heat_map)
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='_predict_bbox_img_with_keypoint.png',
                                                       img=predict_bbox_img_with_keypoint)
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='_exo_rgb_gt.png',
                                                       img=exo_rgb_gt)
                    ims, txts, links = html_append_img(ims, txts, links, batch_i, out_image_folder, name='gt_bbox_img_with_keypoint.png',
                                                       img=gt_bbox_img_with_keypoint)


                    html.add_images(ims, txts, links)
                    html.save()
                    ims = []
                    txts = []
                    links = []

        print('Final Pose accuracy:' + str(pose_correct_count / total_count))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')

    parser.add_argument('--data-config', type=str, default='cfg/person.data', help='path to data config file')
    parser.add_argument('--weights', type=str, default='weight_retina_04_22_Pose_Affordance/backup2.pt', help='path to weights file')
    parser.add_argument('--n-cpus', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--worker', type=str, default='first', help='size of each image dimension')
    parser.add_argument('--out', type=str, default='/home/yangmingwen/first_third_person/first_third_result/pose_affordance_out_422/', help='cfg file path')
    parser.add_argument('--cfg', type=str, default='cfg/rgb-encoder.cfg,cfg/classifier.cfg', help='cfg file path')
    parser.add_argument('--testing_data_mode', type=bool, default=True, help='using testing or training data')
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
        n_cpus=opt.n_cpus,
        out=opt.out,
        testing_data=opt.testing_data_mode,
    )
