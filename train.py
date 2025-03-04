import argparse
import time
import pylab as pl
# Import test.py to get mAP after each epoch
import test

# Utils Usage
from utils_lib.datasets import *
from utils_lib.utils import *
from utils_lib.util import *
from focal_loss import FocalLoss
from torch.optim import lr_scheduler
# Model Definition
from models import *
# from networks.network import First_Third_Net
from networks import *
import networks
DARKNET_WEIGHTS_FILENAME = 'darknet53.conv.74'
DARKNET_WEIGHTS_URL = 'https://pjreddie.com/media/files/{}'.format(DARKNET_WEIGHTS_FILENAME)

# Visualize Way
# python -m visdom.server -p 8399
import visdom
vis = visdom.Visdom(port=8399)


# CUDA_VISIBLE_DEVICES=2 python train.py --batch-size 4 --weight weight_retina_05_09_Pose_Affordance_Third_bp_3 --epochs 100 --data-config cfg/person.data


def train(
        net_config_path,
        data_config_path,
        img_size=416,
        resume=True,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        weights_path='weights',
        report=True,
        multi_scale=False,
        freeze_backbone=False,
        var=0,
        gpu_id='2'
):
    from tensorboardX import SummaryWriter
    tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt.weights_path)

    # device = torch_utils.select_device(gpu_choice=gpu_id)
    # print("Using device: \"{}\"".format(device))
    # criterion = FocalLoss()
    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        torch.backends.cudnn.benchmark = True

    os.makedirs(weights_path, exist_ok=True)

    # Save model path
    latest_weights_file = os.path.join(weights_path, 'latest.pt')
    best_weights_file = os.path.join(weights_path, 'best.pt')

    # Configure run
    data_config = parse_data_config(data_config_path)
    num_classes = int(data_config['classes'])
    train_path = data_config['train']
    pickle_video_mask = data_config['pickle_video_mask_train']
    # pickle_ignore_mask = data_config['pickle_ignore_mask_train']

    # Initialize model
    # here, for the rgb is 416 = 32 by 13 but for the classifier is 13 by 13
    model = networks.network.First_Third_Net()

    # default log
    log_count = 0
    # load_pretained = True
    # if load_pretained:
    #     model.load_state_dict(torch.load('./model/net.pth'))

    print(model)

    # Get dataloader
    dataloader = load_images_and_labels(train_path, batch_size=batch_size, img_size=img_size,
                                        multi_scale=multi_scale, augment=False, center_crop=True,
                                        video_mask=pickle_video_mask)

    lr0 = 0.0001
    resume = True
    if resume:
        latest_weights_file = '/home/yangmingwen/first_third_person/first_third_understanding/weight_retina_10_11_Pose_Classification_TSM?Batch16?_sequebce-cetner1_sgd_shuffle_lr_0.00011/tmp_epo2_2500.pt'
        checkpoint = torch.load(latest_weights_file, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.cuda().train()
        # Set optimizer
        optim_params =[]
        for k, v in model.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_params.append(v)

        # torch.save(self.netG_tmp.state_dict(), '/home/tmp_weight.pth')
        # self.netG_tmp.load_state_dict(torch.load('/home/tmp_weight.pth'))

        # optimizer = torch.optim.Adam(optim_params, lr=lr0)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, momentum=.9)
        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']
        del checkpoint  # current, saved
    else:
        start_epoch = 0
        best_loss = float('inf')
        # TODO: Initialize model with darknet53 weights (optional)
        load_darnet = False
        if load_darnet == True:
            def_weight_file = os.path.join(weights_path, DARKNET_WEIGHTS_FILENAME)
            if not os.path.isfile(def_weight_file):
                os.system('wget {} -P {}'.format(
                    DARKNET_WEIGHTS_URL,
                    weights_path))
            assert os.path.isfile(def_weight_file)
            model.load_weights(def_weight_file)
            print('Init model with Darknet53 training begin >>>>>>')
        else:
            print('Init training begin >>>>>>')
        model.cuda().train()
        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    model_info(model)
    t0 = time.time()
    from utils_lib.utils import VisuaLoss
    visLoss = VisuaLoss(vis)
    count_i = 0
    for epoch in range(epochs):

        epoch += start_epoch
        print(('%8s%12s' + '%10s' * 14) % ('Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R',
                                           'nTargets', 'TP', 'FP', 'FN', 'time'))

        # Update scheduler (automatic)
        # @TODO Trying LR here @yangming
        # scheduler.step()
        # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5


        if epoch >= 3 + start_epoch:
            lr = lr0 / 10
            # mask_loss_switch = True
            # current_mask_loss_switch = mask_loss_switch
            # constain_switch = True

        elif epoch > 10 + start_epoch:
            lr = lr0 / 100
            # mask_loss_switch = True
            # current_mask_loss_switch = mask_loss_switch
            # constain_switch = True

        else:
            lr = lr0
        #
        for g in optimizer.param_groups:
            g['lr'] = lr

        # optimizer.zero_grad()

        # write loss file
        file = open(os.path.join(weights_path, 'weights_results.txt'), 'a')
        file.write(str(weights_path))

        for i, (imgs, targets, scenes, scenes_gt, video_mask, frame_mask) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # print('Current_lr:' + str(lr))

            # Train Pose Branch
            constain_switch = False
            current_mask_loss_switch = True

            # Compute loss, compute gradient, update parameters
            loss = model(imgs, scenes, scenes_gt, targets, video_mask, frame_mask, mask_loss_switch=current_mask_loss_switch, constain_switch=constain_switch)
            loss.backward()

            # drawing_bbox_gt(input=model.exo_rgb, bbox=gt_bbox, label=gt_label, name='gt_', vis=vis)
            # drawing_bbox_gt(input=model.exo_rgb, bbox=predict_bbox, label=predict_label, name='predict_', vis=vis)

            # @TODO: Muilti-batch here
            # accumulate gradient for x batches before optimizing
            # if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):

            count_i += 1

            if count_i >= 2:
                count_i = 0

                log_count = log_count + 1
                # Tensorboard Loge
                tb_logger.add_scalar('pose_loss', float(model.losses['pose_loss']), log_count)
                tb_logger.add_scalar('affordance_loss', float(model.losses['affordance_loss']), log_count)
                tb_logger.add_scalar('mask_loss', float(model.losses['mask_loss']), log_count)
                tb_logger.add_scalar('constrain_loss', float(model.losses['constrain_loss']), log_count)
                tb_logger.add_scalar('total_loss', float(loss), i)

                optimizer.step()
                optimizer.zero_grad()
                s =('%g/%g' % (epoch, epochs - 1),
                           '%g/%g' % (i, len(dataloader) - 1),
                            'total_loss', float(loss),
                            'pose_loss:', float(model.losses['pose_loss']),
                            'affordance_loss', float(model.losses['affordance_loss']),
                            'mask_loss', float(model.losses['mask_loss']),
                            'constrain_loss', float(model.losses['constrain_loss']),
                            'time:', time.time() - t0)
                print(s)
                t0 = time.time()
                # Write epoch results
                file.write(str(s))
                file.write('\n')

            # optimizer.step()
            # optimizer.zero_grad()
            # Running epoch-means of tracked metrics
            # ui += 1
            # for key, val in model.losses.items():
            #     rloss[key] = rloss[key] #  (rloss[key] * ui + val) / (ui + 1)



            # visLoss.plot_current_errors(i, 1, rloss)
            # if loss.detach().cpu().numpy():
            # Update best loss
            # Default NT = 1
            # loss_per_target = rloss['loss'] / rloss['nT']
            # loss_per_target = rloss['loss'] / 1
            # if loss_per_target < best_loss:
            #     best_loss = loss_per_target
            # Save backup weights every 5 epochs



            # if (epoch > 0) & (epoch % 200 == 0):
            #     backup_file_name = 'backup{}.pt'.format(epoch)
            #     backup_file_path = os.path.join(weights_path, backup_file_name)
            #     os.system('cp {} {}'.format(
            #         latest_weights_file,
            #         backup_file_path,
            #     ))
            #
            #     checkpoint = {'epoch': epoch,
            #                   'best_loss': best_loss,
            #                   'model': model.state_dict(),
            #                   'optimizer': optimizer.state_dict()}
            #     torch.save(checkpoint, backup_file_path)
            #     print('Save Model Backup')
            #
            # if i % 500 == 0:
            #     # Save latest checkpoint
            #     checkpoint = {'epoch': i,
            #                   'best_loss': best_loss,
            #                   'model': model.state_dict(),
            #                   'optimizer': optimizer.state_dict()}
            #     print('Save Model_latest')
            #     tmp_weights_file = os.path.join(weights_path, 'tmp'+str(i)+'.pt')
            #     torch.save(checkpoint, latest_weights_file)
            #
            if (i > 1) & (i % 2500 == 0):
                # Save tmp checkpoint
                checkpoint = {'epoch': i,
                              'best_loss': best_loss,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                print('Save Model_latest')
                tmp_weights_file = os.path.join(weights_path, 'tmp_epo'+str(epoch) + '_' +str(i)+'.pt')
                torch.save(checkpoint, tmp_weights_file)

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, best_weights_file)
        print('Save Model Best')

        # file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--data-config', type=str, default='cfg/coco.data', help='path to data config file')
    parser.add_argument('--cfg', type=str, default='cfg/rgb-encoder.cfg,cfg/classifier.cfg', help='cfg file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img_size_extra', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--weights-path', type=str, default='weight_overfit_one_frame_2_1',
                        help='path to store weights')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--report', action='store_true', help='report TP, FP, FN, P and R per batch (slower)')
    parser.add_argument('--freeze', action='store_true', help='freeze darknet53.conv.74 layers for first epoch')
    parser.add_argument('--var', type=float, default=0, help='optional test variable')
    parser.add_argument('--gpu_id', type=str, default='3', help='optional gpu variable')
    parser.add_argument('--worker', type=str, default='first', help='detection or first-person video understand')

    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()
    torch.cuda.empty_cache()

    train(
        opt.cfg,
        opt.data_config,
        img_size=opt.img_size_extra,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        weights_path=opt.weights_path,
        report=opt.report,
        multi_scale=opt.multi_scale,
        freeze_backbone=opt.freeze,
        var=opt.var,
        gpu_id=opt.gpu_id
    )
