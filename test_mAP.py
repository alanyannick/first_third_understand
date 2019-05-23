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

    total_y_true = []
    total_y_score = []
    if scene_flag:
        for batch_i, (imgs, targets, scenes, scenes_gt, video_mask, frame_mask) in enumerate(dataloader):
            total_count += 1
            if batch_i > 10000000:
                break
            with torch.no_grad():
                if worker == 'detection':
                    output = model(imgs.cuda())
                else:

                    pose_label, pose_affordance, frame_affordance= model(imgs, scenes, scenes_gt, targets, video_mask, frame_mask, test_mode=True)
                    predict_pose_label = pose_label.cpu().float().numpy()[0]
                    gt_pose_label = np.array(targets[0][0][0])

                    # Bilinear version
                    # sem_frame_affordance_mask = torch.argmax(F.interpolate(frame_affordance.permute(0, 3, 1, 2), size=(800, 800), mode='bilinear').
                                                           #  permute(0, 2, 3, 1), dim=3).cpu().float().numpy()[0] + 24

                    # Create the frame mask here
                    mask_different_class = []
                    mask_for_map = torch.argmax(frame_affordance, dim=3).cpu().float().numpy()[0]

                    # Assuming we only have one class & we want to get the final 1pose index of 0,1 mask ( out of 7 index mask)
                    for i in range(0, 7):
                        mask_different_class.append(((mask_for_map == i)).astype(int))

                        if ((mask_for_map == i)).astype(int).max() >= 1 :
                            calculate_predict = i

                     # Get the X pose predicition and corresponding index of the X pose in GT mask
                    frame_affordance_video = mask_soft_max(frame_affordance)
                    frame_heat_affordance = frame_affordance_video[:, :, :, calculate_predict].clone().squeeze(
                            0).cpu().float().numpy()
                    index = frame_heat_affordance == frame_heat_affordance.max()

                    # Get the center of the prediction with the highest confidence
                    x_center, y_center = np.where(index.astype(int) == 1)
                    # Transfer to the center
                    x_center = x_center[0] + 0.5
                    y_center = y_center[0] + 0.5

                    # Transform targets
                    x1 = (targets[0][0][1]/2 - targets[0][0][3] / 4) * 13
                    y1 = (targets[0][0][2]/2 - targets[0][0][4] / 4) * 13
                    x2 = (targets[0][0][1]/2 + targets[0][0][3] / 4) * 13
                    y2 = (targets[0][0][2]/2 + targets[0][0][4] / 4) * 13

                    # Get the GT region
                    # x1 = (targets[0][0][1] - targets[0][0][3] / 2) * 13
                    # y1 = (targets[0][0][2] - targets[0][0][4] / 2) * 13
                    # x2 = (targets[0][0][1] + targets[0][0][3] / 2) * 13
                    # y2 = (targets[0][0][2] + targets[0][0][4] / 2) * 13

                    # Create corresponding X - mAP Map, other region we will set to the 0 value.
                    y_true = np.zeros((7,13,13), dtype=np.float16)
                    if x1 < x_center < x2 and y1 < y_center < y2:
                        print('Hit')
                        # Set the region
                        y_true[calculate_predict][int(x1):int(x2), int(y1):int(y2)] = 1

                    y_score = frame_affordance_video[:, :, :, :7].squeeze(
                            0).permute(2,0,1).cpu().float().numpy()

                    total_y_true.append(y_true)
                    total_y_score.append(y_score)

        total_y_true = np.array(total_y_true)
        total_y_score = np.array(total_y_score)

        for each_class in range(0, 7):
            per_y_score = total_y_score[:total_y_score.shape[0], each_class, :, :].reshape(total_y_score.shape[0] * 13 * 13)
            per_y_true = total_y_true[:total_y_true.shape[0], each_class, :, :].reshape(total_y_score.shape[0] * 13 * 13)
            # predict:
            print('Class' + str(each_class) + '_AP: ')
            # calculate MAP
            print(average_precision_score(per_y_true, per_y_score))


        print('Final Pose accuracy:' + str(pose_correct_count / total_count))
        class_balance_accuracy = np.array(corrct_class_balance) / np.array(total_classes)
        print('Final Balance Class Pose accuracy:' + str(class_balance_accuracy.mean()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')

    parser.add_argument('--data-config', type=str, default='cfg/person.data', help='path to data config file')
    parser.add_argument('--weights', type=str, default='weight_retina_05_21_Pose_Affordance_Third_Final_Version_Loss_Switch_warm_up_2400/tmp_epo2_2500.pt', help='path to weights file')
    parser.add_argument('--n-cpus', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--worker', type=str, default='first', help='size of each image dimension')
    parser.add_argument('--out', type=str, default='/home/yangmingwen/first_third_person/first_third_result/weight_retina_05_24_Pose_Affordance_Third_Final_Version_Equal_Loss_pretrain/', help='cfg file path')
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
