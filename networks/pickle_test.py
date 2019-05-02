import os
import pickle
import collections, numpy
frame_mask='/home/yangmingwen/first_third_person/merged_clusters/final_branch_gt_merged.pickle'

total_count = 0
count_bg = 0
count_0 = 0
count_1= 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
count_6 = 0
count_7 = 0

with open(frame_mask, 'rb') as gt_handle:
    gt_per_frame_mask = pickle.load(gt_handle)
    for index in gt_per_frame_mask:
        # print(gt_per_frame_mask[index])
        # unique, counts = numpy.unique(gt_per_frame_mask[index], return_counts=True)
        # dict(zip(unique, counts))
        count_0 += numpy.count_nonzero(gt_per_frame_mask[index] == 0)
        count_1 += numpy.count_nonzero(gt_per_frame_mask[index] == 1)
        count_2 += numpy.count_nonzero(gt_per_frame_mask[index] == 2)
        count_3 += numpy.count_nonzero(gt_per_frame_mask[index] == 3)
        count_4 += numpy.count_nonzero(gt_per_frame_mask[index] == 4)
        count_5 += numpy.count_nonzero(gt_per_frame_mask[index] == 5)
        count_6 += numpy.count_nonzero(gt_per_frame_mask[index] == 6)
        count_bg += numpy.count_nonzero(gt_per_frame_mask[index] == -1)

        total_count += numpy.count_nonzero(gt_per_frame_mask[index] < 8)
        # print(numpy.count_nonzero(gt_per_frame_mask[index] == 1))
        # print(numpy.count_nonzero(gt_per_frame_mask[index] < 8))
    print('weight: from 0 - -1')
    print(count_0/total_count)
    print('  ')
    print(count_2/total_count)
    print('  ')
    print(count_3/total_count)
    print('  ')
    print(count_4/total_count)
    print('  ')
    print(count_5/total_count)
    print('  ')
    print(count_6/total_count)
    print('  ')
    print(count_bg/total_count)
    print('  ')
        # print