import os
import numpy as np 
from PIL import Image
from tools.dataset import Dataset
import torch

def read_images_paths(dataset_dir_list):
    class_names = sorted([x for data_set in dataset_dir_list 
                            for x in os.listdir(data_set)
                                  if os.path.isdir(os.path.join(data_set, x))])

    res = np.array(class_names) 
    class_names = np.unique(res) 

    image_files = [[[os.path.join(data_set, class_name, x)
                    for x in os.listdir(os.path.join(data_set, class_name))]
                    for data_set in dataset_dir_list]
                    for class_name in class_names]

    image_file_list = []
    image_label_list = []

    for i, class_name in enumerate(class_names): 
        for j, data_set in enumerate(dataset_dir_list):
            image_file_list.extend(image_files[i][j])
            image_label_list.extend([i] * len(image_files[i][j]))

    image_width, image_height = Image.open(image_file_list[0]).size

    return image_file_list, image_label_list, image_width, image_height, class_names


def split_dataset(dataset_dirs, validation_split):
    

    image_file_list, image_label_list, image_width, image_height, class_names = read_images_paths(dataset_dirs)

    total_images = len(image_file_list)

    indices = list(range(total_images))
    split = int(np.floor(validation_split * total_images))

    np.random.shuffle(indices)
        
    train_indices, val_indices = indices[split:], indices[:split]

    train_images, train_labels = list(), list()
    vall_images, vall_labels = list(), list()


    for i in train_indices:
        train_images.append(image_file_list[i])
        train_labels.append(image_label_list[i])


    for i in val_indices:
        vall_images.append(image_file_list[i])
        vall_labels.append(image_label_list[i])

    train_labels_count = [train_labels.count(i) for i in range(len(class_names))]
    vall_labels_count = [vall_labels.count(i) for i in range(len(class_names))]

    return (train_images, train_labels, train_labels_count, 
                 vall_images, vall_labels, vall_labels_count, 
                 image_width, image_height, class_names)