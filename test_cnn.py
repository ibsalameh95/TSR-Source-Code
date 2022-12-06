import torch
import numpy as np
import argparse
import pandas as pd
import torch.utils.data
from tools.dataset import Dataset
from tqdm import tqdm
from tools.prepare_dataset import read_images_paths
from tools.data_visualization import plot_confusion_matrix
from tools.load_model import get_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.image as mpimg

parser = argparse.ArgumentParser(
    description='Train a deep MIL using VGG19 using Transfar Learning')


# Arguments to be provided before running the code.
parser.add_argument('--test_dataset_dir', default='data/TSR_MY_TEST',
                    help='test dataset path', dest='test_dataset_dir')   
parser.add_argument('--trained_model_path', default='./trained_models/CRC-HE-CLASS.pth',
                    help='trained model path', dest='trained_model_path')  
parser.add_argument('--batch_size', default='4', type=int,
                    help='batch size', dest='batch_size')



FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)


def test_model(model, device, data_loader, class_names):  
    total, running_accuracy = 0, 0 
    y_true = list()
    y_pred = list()
    image_paths = list()
    
    test_pbar = tqdm(total=len(data_loader))

    with torch.no_grad():
        for paths, images, labels in data_loader:
            test_pbar.update(1)

            inputs, outputs = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            predicted_outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(predicted_outputs, 1)

            total += outputs.size(0) 
            running_accuracy += (predicted == outputs).sum().item() 

            for i in range(len(outputs)):
                image_paths.append(paths[i])
                y_true.append(class_names[outputs[i]])
                y_pred.append(class_names[predicted[i]])

        testing_results_dic = {'image_path': image_paths, 'y_true': y_true, 'y_pred': y_pred}
        testing_results = pd.DataFrame(testing_results_dic, columns=['image_path', 'y_true', 'y_pred'])
        test_pbar.close()

        print('Accuracy of the model based on the test set of ', len(data_loader) ,' inputs is: %d %%' % (100 * running_accuracy / total))    
    return testing_results

test_vall_dasets_dirs = '' 

random_seed = 42

np.random.seed(random_seed)

print('Preparing testing datasets: ...')

if FLAGS.test_dataset_dir != '':
    dataset_type = 'test' 
    dataset_dir = [FLAGS.test_dataset_dir]
    test_images, test_labels, image_width, image_height, class_names = read_images_paths(dataset_dir)
    test_labels_count = [test_labels.count(i) for i in range(len(class_names))]

print('# Testing dataset summary:')
print('# Total images = {}'.format(len(test_images)))
print('# Image dimentions = {} * {}'.format(image_width, image_height))
print('# Label names = {}'.format(class_names))
print('# Label counts = {} \n'.format(test_labels_count))

test_dataset = Dataset('test', test_images, test_labels)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size= FLAGS.batch_size, num_workers= 0, shuffle= "False", drop_last= False)


model, device = get_model(len(class_names))
model.load_state_dict(torch.load(FLAGS.trained_model_path))

print('cuda is available: {} \n'.format(torch.cuda.is_available()))
print('moving model to available device')
model.to(device)


print('# Model Hyperparameters: ')
print('batch size = {} \n'.format(FLAGS.batch_size))


# start training and evaluation
testing_results = test_model(model, device, test_dataloader, class_names)
testing_results.to_csv('./outputs/files/test_results.csv', header= True)


# plot testing confusion matrix

data_arr = pd.read_csv('./outputs/files/validation_results.csv')

label_index_arr = np.asarray(data_arr['y_true'],dtype=str)
pred_arr = np.asarray(data_arr['y_pred'],dtype=str)

conf_mat = confusion_matrix(label_index_arr, pred_arr)

fig, ax = plt.subplots(figsize=(10,10))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=True, current_ax=ax)
fig.subplots_adjust(left=0.15, bottom=0.28, right=0.94, top=0.99, wspace=0.2 ,hspace=0.20 )

plt.savefig('./outputs/files/testing_results_cm.png')


# plot wrongly predicted images: 

df = (pd.read_csv('./outputs/files/test_results.csv')[lambda x: x['y_true'] != x['y_pred']])

num_row = 9
num_col = 4
images = {}
classes = df['y_true'].unique()

for i in range(len(classes)):
    images[classes[i]] = np.random.choice(df['y_true'][lambda x: x == classes[i]].index.tolist(),num_col).tolist()

_, axs = plt.subplots(num_row, num_col, figsize=(10, 20))
_.tight_layout(pad=2)
for i,c in enumerate(classes): 
    for img, ax in zip(images[c], axs[i]):
        ax.imshow(mpimg.imread(df['image_path'][img]))
        ax.set_title('{} -> {}'.format(df['y_true'][img], df['y_pred'][img]), loc= 'left')

plt.savefig('./outputs/files/wrongly_predicted_images_test_ds.png')
