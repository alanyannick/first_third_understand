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


out_file_link_balance = '/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/file_list_test_new_4_22_video_demo.txt'
number_frames = 300
source_folder ='/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/images/'


with open(out_file_link_balance, 'a') as gt_file:
    with open(list_file) as f:
        source_link = f.readlines()
        for i in range(0, len(datasets_list)):
            for index in range(0, number_frames):
                # '5UHV3EGO_5_first-00669.jpg

                img_index_iter = str(index)
                init_index = 5 - len(str(index))
                final_index_name = list('00000')
                start_index = 0
                for index in range(init_index, 5):
                    final_index_name[index] = img_index_iter[start_index]
                    start_index += 1
                final_replace_name = ''.join(str(e) for e in final_index_name)

                link = source_folder + datasets_list[i] + 'first-' + final_replace_name + '.jpg' +'\n'
                gt_file.write(link)

gt_file.close()
