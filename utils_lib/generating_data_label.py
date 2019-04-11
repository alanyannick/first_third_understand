import torch
import os
# create the reference link here
# Input file
label_file_path = '/home/yangmingwen/first_third_person/data_2019_3_15/train_pose_balanced_cluster_removed/darknet_targets_train.pth'
tsv_file_path = '/home/yangmingwen/first_third_person/data_2019_3_15/train_pose_balanced_cluster_removed/meta_train.tsv'

# Save dir
out_file_dir = '/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/labels/'
# Write file_list dir
image_file_dir = '/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/images/'
scene_file_dir = '/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/scenes'

# Output: 1.Get the list of images 2. Get the txt file contained GT
list_file = open('/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/file_list_train_new.txt', 'w')

cxcywh2xyxy = False

# load model to transfer pth gt to mscoco name_file.txt gt
darknet_targets = torch.load(label_file_path).type(torch.float)
with open(tsv_file_path) as f:
    source_link = f.readlines()
    for index in range(0, len(source_link)):
        # get the link
        link = source_link[index]

        # get the ego_link'
        image_name = link.split('\t')[0].split('/')[-1].split('.jpg')[0]
        ego_link = image_name + '.txt'
        out_file_link = out_file_dir + ego_link

        # get the gt: cls, x_center, y_center, w, h
        ground_truth = darknet_targets[index].tolist()

        # create the final output txt file here.
        gt_file = open(out_file_link, 'w')

        # ValueError: could not convert string to float: '3.0,'
        # notice here should be float, rather than when reads the file you will meet this problems
        for num in ground_truth:
            gt_file.write(str((num)) + ' ')
        gt_file.write('\n')
        gt_file.close()
        # ----- previous => Write label.txt to label file ---

        # verify the scene link
        exo_scene_link = link.split('\t')[1].split('/')[-1]
        ego_view_link = link.split('\t')[0].split('/')[-1]
        exo_out_link = scene_file_dir + exo_scene_link
        ego_out_link = image_file_dir + ego_view_link
        # get the final listfile
        # list_file.write(ego_out_link + ' ' + exo_out_link + '\n')
        list_file.write(ego_out_link + '\n')





