import torch
import argparse
import os.path
import os
import torch.utils.data
from tools.load_model import TransfereLearning as TS
import matplotlib.pyplot as plt
import openslide 
import imageio
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from tools import prediction2mask

parser = argparse.ArgumentParser(
    description='Testing deep MIL using VGG19 using Transfar Learning')


# Arguments to be provided before running the code.
parser.add_argument('--wsi_dir', default='WSI/segmentatiom_test',
                    help='WSI info folder', dest='wsi_dir')
parser.add_argument('--stride', default=84, type=int,
                    help='batch size', dest='stride')
parser.add_argument('--window_height', default=224, type=int,
                    help='sliding window height', dest='window_height')
parser.add_argument('--window_width', default=224, type=int,
                    help='sliding window width', dest='window_width')                    
parser.add_argument('--read_level', default=1, type=int,
                    help='1 for 20x magnification', dest='read_level')  


FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)


# get number of classes from the dataset directories
classes = os.walk('data/TSR_MY_TEST').__next__()[1]


trained_model_path = 'trained_model/CRC-HE-CLASS_2.pth'

if not os.path.exists(os.path.join('files/cropped_tiles/level0')):
    os.makedirs(os.path.join('files/cropped_tiles/level0'))

if not os.path.exists(os.path.join('files/cropped_tiles/level1')):
    os.makedirs(os.path.join('files/cropped_tiles/level1'))    

if not os.path.exists(os.path.join('files/cropped_tiles/level1_segmented')):
    os.makedirs(os.path.join('files/cropped_tiles/level1_segmented'))  

if not os.path.exists(os.path.join('WSI/segmented_slide')):
    os.makedirs(os.path.join('WSI/segmented_slide'))  


img_transforms = transforms.Compose([transforms.ToTensor()])

#initialize the model
model, device = TS.load_model(len(classes))
trained_model = torch.load(os.path.join(trained_model_path), map_location=torch.device('cpu'))

model.load_state_dict(trained_model)

with open('files/prediction_info.txt', 'w') as f: 
    f.write('row_read \t col_read \t img_num \t predicion \n')

with open('files/prediction_info.txt', 'a') as prediction_info:
    slide_name = ''
    for path, dirnames, filenames in os.walk(FLAGS.wsi_dir):
        for file in filenames:
            slide_name = file

    slide_path = os.path.join(FLAGS.wsi_dir, slide_name)
    slide = openslide.OpenSlide(slide_path)

    print(slide.level_downsamples)
    print(slide.level_dimensions)

    
    factor = round(slide.level_dimensions[0][0] / slide.level_dimensions[1][0])

    read_level = FLAGS.read_level
    stride = FLAGS.stride

    window_width = FLAGS.window_width
    window_height = FLAGS.window_height
    

    #Level 1 image dimentions
    if int(slide.properties.get('aperio.AppMag')) == 40 and factor == 2: 
        image_w, image_h = slide.level_dimensions[1][0], slide.level_dimensions[1][1]

    if int(slide.properties.get('aperio.AppMag')) == 40 and factor != 2:
        image_w, image_h = slide.level_dimensions[0][0]/2, slide.level_dimensions[0][1]/2

    if int(slide.properties.get('aperio.AppMag')) == 20: 
        image_w, image_h = slide.level_dimensions[0][0], slide.level_dimensions[0][1]

    masked_slide = torch.zeros(3, int(image_w), int(image_h))
    
    pbar = tqdm(total=int(image_h/stride) * int(image_w/stride))
    
    count = 0

    for col in range(1, int((image_h/stride)/factor) - 1):
        for row in range(1, int((image_w/stride)/factor) - 1):
            pbar.update(1)

            col_read = col * stride * factor # H
            row_read = row * stride * factor # W

            image = slide.read_region((row_read,col_read), read_level, (window_width,window_height)).convert('RGB')
            #image2 = slide.read_region((row_read,col_read), 0, (window_width,window_height)).convert('RGB')  

            im_arr = np.array(image)[:, :, 0:3]
            #im_arr2 = np.array(image2)[:, :, 0:3]

            count += 1

            # im_arr_avg = np.mean(im_arr, axis=2)  # mean of RGB values combined per pixel
            # im_arr_avg_bool = im_arr_avg < 10 # avg colors not black 
            # black_ratio = np.sum(im_arr_avg_bool)/np.square(window_width)

            # if black_ratio > 0.90 or (image.height < window_height or image.width < window_width):
            #     continue
            # else: 

            imageio.imwrite('files/cropped_tiles/level1/'+ str(count) + '.jpeg', im_arr)
            #imageio.imwrite('files/cropped_tiles/level0/'+ str(count) + '.jpeg', im_arr2)
            image = img_transforms(image)
            n_channels, width, height = image.shape

            image = image.reshape(1,n_channels, width, height)
            with torch.no_grad():
                # calculate outputs by running images through the network
                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)

                prediction_info.write('{} \t {} \t {} \t {} \n'.format(row_read, col_read , str(count), classes[predicted]))
                # convert prediction to color
                masked_tile = prediction2mask.image_to_mask(image, classes[predicted])
                im_arr_masked = np.array(masked_tile)[:, :, 0:3]

                trans = transforms.PILToTensor()
                masked_tile = trans(masked_tile)

                imageio.imwrite('files/cropped_tiles/level1_segmented/'+ str(count) + '.jpeg', im_arr_masked)
                masked_slide[:, row: row + window_width, col: col + window_height] = masked_tile

                outfile = 'files/masked_slide.tif'


    masked_slide_arr = np.array(masked_slide)[:, :, 0:3]
    imageio.imwrite('WSI/segmented_slide/'+ slide_name + '.tif', masked_slide_arr)
    
    #prediction2mask.tissue_class_ratio(masked_slide_arr, 9, True)

    pbar.close()                    