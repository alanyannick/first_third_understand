list_file = '/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/file_list_train_new_4_22.txt'
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


out_file_link_balance = '/home/yangmingwen/first_third_person/data_2019_3_15/final-dataset/file_list_train_new_4_22_verify.txt'
with open(out_file_link_balance, 'a') as gt_file:
    with open(list_file) as f:
        source_link = f.readlines()
        for i in range(0, len(datasets_list)):
            for index in range(0, len(source_link)):
                link = source_link[index]
                if datasets_list[i] in link:
                    gt_file.write(link)
                    if index > 700:
                        break
                else:
                    continue


gt_file.close()
