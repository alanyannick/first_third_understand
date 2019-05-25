import torch
import os
import cv2
import numpy as np

# create the reference link here
label_file_path = '/home/yangmingwen/first_third_person/data_2019_3_15/merge/cluster_merge_test/darknet_targets_test_level1.pth'
tsv_file_path = '/home/yangmingwen/first_third_person/data_2019_3_15/merge/cluster_merge_test/meta_test.tsv'
# read tsv
with open(tsv_file_path) as tsvfile:
    meta = [line.rstrip().split('\t') for line in tsvfile.readlines()]
#load pth
darknet_targets = torch.load(label_file_path).type(torch.float)
# random
my_perm = np.random.permutation(len(meta))
meta = [meta[i] for i in my_perm]
darknet_targets = darknet_targets[my_perm.astype(int), ...]

# Input file
label_file_path = '/home/yangmingwen/first_third_person/data_2019_3_15/merge/cluster_merge_test/darknet_targets_test_level1.pth'
tsv_file_path = '/home/yangmingwen/first_third_person/data_2019_3_15/merge/cluster_merge_test/meta_test.tsv'

# update random ppth and tsv
torch.save(darknet_targets, label_file_path)
with open(tsv_file_path, 'w' ) as metaout:
    for tmp in meta:
          image_paths = '\t'.join(tmp)
          metaout.write('%s\n' % image_paths)


# Save dir
outdir = '/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/loc_balanced_nips/'
filter_num = 1

# =========== Begin filtering ======================
out_file_link = outdir + 'distribution.txt'
out_file_link_balance = outdir + 'distribution_balance.txt'
# create the distrubute matrix
grid = 13
datasets_distribute = torch.zeros(grid, grid)
datasets_list = []

with open(tsv_file_path) as f:
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


# load model to transfer pth gt to mscoco name_file.txt gt

# restrct the bounding line
darknet_targets[:, 1:][darknet_targets[:, 1:] >= 1] = 0.999
darknet_targets[:, 1:][darknet_targets[:, 1:] <= 0] = 0

with open(tsv_file_path) as f:
    source_link = f.readlines()
    for index in range(0, len(source_link)):
        # get the link
        link = source_link[index]
        # get the ego_link
        image_name = link.split('\t')[0].split('/')[-1].split('.jpg')[0]
        ego_link = image_name + '.txt'
        # get the gt: cls, x_center, y_center, w, h
        ground_truth = darknet_targets[index].tolist()
        ground_truth_x_center = int(ground_truth[1] * grid)
        ground_truth_y_center = int(ground_truth[2] * grid)
        datasets_distribute[ground_truth_x_center, ground_truth_y_center] += 1

    with open(out_file_link, 'w') as gt_file:
        gt_file.write('data distribution shows:' + '\n')
        np.savetxt(gt_file, datasets_distribute.cpu().numpy(), fmt='%4d')
        gt_file.close()


train_meta = []
train_ind = []
datasets_distribute = torch.zeros(grid, grid)
with open(tsv_file_path) as f:
    source_link = f.readlines()
    datasets_distribute_temp = torch.zeros(grid, grid)
    for data_name in datasets_list:
        for index in range(0, len(source_link)):
            # get the link
            link = source_link[index]
            # get the ego_link
            image_name = link.split('\t')[0].split('/')[-1].split('.jpg')[0]
            if data_name in image_name:
                # get the gt: cls, x_center, y_center, w, h
                ground_truth = darknet_targets[index].tolist()
                ground_truth_x_center = int(ground_truth[1] * grid)
                ground_truth_y_center = int(ground_truth[2] * grid)

                if int(datasets_distribute_temp[ground_truth_x_center, ground_truth_y_center].cpu().numpy()) >= filter_num:
                    continue
                else:
                    datasets_distribute_temp[ground_truth_x_center, ground_truth_y_center] += 1
                    train_meta.append(link)
                    train_ind.append(index)
        with open(out_file_link_balance, 'a') as gt_file:
            gt_file.write('balance data distribution shows:' + '\n')
            gt_file.write('video name:' + str(data_name) + '\n')
            np.savetxt(gt_file, datasets_distribute_temp.cpu().numpy(), fmt='%4d')
            datasets_distribute += datasets_distribute_temp
            datasets_distribute_temp = torch.zeros(grid, grid)


with open(out_file_link_balance, 'a') as gt_file:
    gt_file.write('balance data distribution shows:' + str(datasets_distribute.sum().cpu().numpy()) + '\n')
    np.savetxt(gt_file, datasets_distribute.cpu().numpy(), fmt='%4d')
    gt_file.close()
    darknet_targets_train = darknet_targets[np.array(train_ind).astype(int), ...]
    torch.save(darknet_targets_train, os.path.join(outdir, 'balance_loc_darknet_targets_test_random.pth'))
    with open(os.path.join(outdir, 'balance_loc_meta_test_random.tsv'), 'w') as metaout:
        for tmp in train_meta:
            metaout.write(tmp)



