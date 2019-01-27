import torch
import os
# create the reference link here
label_file_path = '/home/yangmingwen/first_third_person/first_third_understanding/data/datasets/darknet_targets.pth'
tsv_file_path = '/home/yangmingwen/first_third_person/first_third_understanding/data/datasets/pruned-meta.tsv'
out_file_dir = '/home/yangmingwen/first_third_person/first_third_understanding/data/datasets/labels/'

# load model
darknet_targets = torch.load(label_file_path).type(torch.float)
with open(tsv_file_path) as f:
    source_link = f.readlines()
    for index in range(0,len(source_link)):
        # get the link
        link = source_link[index]
        # get the ego_link
        ego_link = link.split('\t')[0].split('/')[-1].split('.jpg')[0] + '.txt'
        out_file_link = out_file_dir + ego_link
        # get the gt: cls, x_center, y_center, w, h
        ground_truth = darknet_targets[0].tolist()
        # create the final output txt file here.
        gt_file = open(out_file_link, 'w')
        # ValueError: could not convert string to float: '3.0,'
        # notice here should be float, rather than when reads the file you will meet this problems
        for num in ground_truth:
            gt_file.write(str((num)) + ' ')
        gt_file.write('\n')
        gt_file.close()

# Create the final file_list
path = '/home/yangmingwen/first_third_person/first_third_understanding/data/datasets/images/'
list_file = open('/home/yangmingwen/first_third_person/first_third_understanding/data/datasets/file_list_train.txt', 'w')
for image_path in os.listdir(path):
    list_file.write(os.path.join(path,image_path) + '\n')
list_file.close()


