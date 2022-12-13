import torch
import os
import numpy as np
import torch.nn as nn
import argparse
import pandas as pd
import torch.utils.data
from tools.dataset import Dataset
from tqdm import tqdm
from tools.prepare_dataset import split_dataset
from tools.load_model import get_model
from tools.data_visualization import plot_confusion_matrix
from tools.data_visualization import plt_charts
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(
    description='Train a deep MIL using VGG19 using Transfar Learning')

# Arguments to be provided before running the code.
parser.add_argument('--train_dataset_dir', default='data/TSR-CRC-Training-set-part1',
                    help='dataset info folder', dest='train_dataset_dir')
parser.add_argument('--vall_dataset_dir', default='data/TSR_MY_TEST',
                    help='dataset info folder', dest='vall_dataset_dir')                  
parser.add_argument('--batch_size', default='64', type=int,
                    help='batch size', dest='batch_size')
parser.add_argument('--learning_rate', default='3e-4', type=float,
                    help='number of patches each patient has', dest='learning_rate')
parser.add_argument('--validation_split', default=0.20, type=float,
                    help='validation split ratio', dest='validation_split')   
parser.add_argument('--num_epochs', default=10, type=int,
                    help='number of steps of execution (default: 1000000)', dest='num_epochs')


FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)



if not os.path.exists('./outputs'):
    os.mkdir('./outputs')
if not os.path.exists('./outputs/files'):
    os.mkdir('./outputs/files')   
if not os.path.exists('./outputs/trained_models'):
    os.mkdir('./outputs/trained_models')

# Strat Training.
def train_model(model, device, optimizer, criterion, data_loader):

    train_loss, train_accuracy, total = 0, 0, 0

    train_pbar = tqdm(total=len(data_loader))

    model.train()

    for paths, images, labels in data_loader:

        train_pbar.update(1) # progress par.

        inputs, outputs = images.to(device), labels.to(device) # get the input and real species as outputs;
        optimizer.zero_grad()  # zero the parameter gradients 
        predicted_outputs = model(inputs) # predict output from the model
        loss = criterion(predicted_outputs, outputs) # calculate loss for the predicted output  
        loss.backward() # backpropagate the loss 
        optimizer.step()  # adjust parameters based on the calculated gradients 
        train_loss += loss.item()  # track the loss value 
        total += outputs.size(0)  # track number of predictions
        _, predicted = torch.max(predicted_outputs, 1) # The label with the highest value will be our prediction 
        train_accuracy += (predicted == outputs).sum().item() # number of matched predictions

    train_pbar.close()
    # Calculate loss as the sum of loss in each batch divided by the total number of predictions done.  
    train_loss = round(train_loss/len(data_loader),2)
    # Calculate accuracy as the number of correct predictions in the batch divided by the total number of predictions done.  
    train_accuracy = round((100 * train_accuracy /total),2)

    return model, train_loss, train_accuracy


def validate_model(model, device, data_loader, class_names):  
    model.eval()

    image_paths, y_true, y_pred = list(), list(), list()

    vall_loss, vall_accuracy, total = 0, 0, 0

    validate_pbar = tqdm(total=len(data_loader))

    with torch.no_grad():
        for paths, images, labels in data_loader:
            validate_pbar.update(1)

            inputs, outputs = images.to(device), labels.to(device)

            predicted_outputs = model(inputs)

            loss = criterion(predicted_outputs, outputs)

            total += outputs.size(0)

            vall_loss += loss.item()

            _, predicted = torch.max(predicted_outputs, 1)

            vall_accuracy += (predicted == outputs).sum().item()

            for i in range(len(outputs)):
                image_paths.append(paths[i])
                y_true.append(class_names[outputs[i]])
                y_pred.append(class_names[predicted[i]])

        validate_pbar.close() 

        vall_loss = round(vall_loss/len(data_loader), 2)
        vall_accuracy = round((100*vall_accuracy/ total),2)

    validation_results_dic = {'image_path': image_paths, 'y_true': y_true, 'y_pred': y_pred}
    validation_results = pd.DataFrame(validation_results_dic, columns=['image_path', 'y_true', 'y_pred'])

    return vall_loss, vall_accuracy, validation_results    


def train_val_model(model, device, optimizer, criterion, train_loader, vall_loader, class_names, num_epochs):
    train_loss_list = []
    train_accuracy_list = []
    vall_loss_list = []
    vall_accuracy_list = []
    
    best_accuracy = 0 
    
    for epoch in range(num_epochs):
        
        print('############## EPOCH - {} ##############'.format(epoch+1))
        
        # train model
        print('******** training ******** \n')      
        
        model, train_loss, train_accuracy = train_model(model, device, optimizer, criterion, train_loader)
        print('tarin_loss= {:.2f}, train_accuracy= {:.2f}% \n'.format(train_loss, train_accuracy))

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        
        
        # validate model
        print('******** validating ******** \n')
        vall_loss, vall_accuracy, validation_results = validate_model(model, device, vall_loader, class_names)
        print('vall_loss= {:.2f}, vall_accuracy= {:.2f}% \n'.format(vall_loss, vall_accuracy))

        vall_loss_list.append(vall_loss)
        vall_accuracy_list.append(vall_accuracy)
        
        if vall_accuracy > best_accuracy:
            best_accuracy = vall_accuracy
            print('saving model with best accuracy of: {:.2f}% \n'.format(best_accuracy))
            torch.save(model.state_dict(), './outputs/trained_models/CRC-HE-CLASS.pth')
            
    # saving data into csv        
    train_data = pd.DataFrame({'loss': train_loss_list, 'accuracy': train_accuracy_list}, columns= ['loss', 'accuracy'])
    vall_data = pd.DataFrame({'loss': vall_loss_list, 'accuracy': vall_accuracy_list}, columns= ['loss', 'accuracy'])
    
    train_data.to_csv('./outputs/files/train_data.csv', header= True)
    vall_data.to_csv('./outputs/files/vall_data.csv', header= True)
    
    validation_results.to_csv('./outputs/files/validation_results.csv', header= True)



 

random_seed = 42

np.random.seed(random_seed)

print('Preparing datasets: ...')

if FLAGS.train_dataset_dir != '' and FLAGS.vall_dataset_dir != '':
    dataset_dirs = [FLAGS.train_dataset_dir, FLAGS.vall_dataset_dir]
    (train_images, train_labels, train_labels_count, vall_images, 
    vall_labels, vall_labels_count, image_width, image_height, class_names) = split_dataset(dataset_dirs, FLAGS.validation_split)

    print('# Training dataset summary:')
    print('# Total images = {}'.format(len(train_images)))
    print('# Image dimentions = {} * {}'.format(image_width, image_height))
    print('# Label names = {}'.format(class_names))
    print('# Label counts = {} \n'.format(train_labels_count))

    print('# Validation dataset summary:')
    print('# Total images = {}'.format(len(vall_images)))
    print('# Image dimentions = {} * {}'.format(image_width, image_height))
    print('# Label names = {}'.format(class_names))
    print('# Label counts = {} \n'.format(vall_labels_count))


train_dataset = Dataset('train', train_images, train_labels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= FLAGS.batch_size, num_workers= 0, shuffle= "True", drop_last= False)

vall_dataset = Dataset('validation', vall_images, vall_labels)
vall_dataloader = torch.utils.data.DataLoader(vall_dataset, batch_size= FLAGS.batch_size, num_workers= 0, shuffle= "False", drop_last= False)


model, device = get_model(len(class_names))

print('cuda is available: {} \n'.format(torch.cuda.is_available()))
print('moving model to available device')
model.to(device)
        
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= FLAGS.learning_rate, momentum=0.9)

print('# Model Hyperparameters: ')
print('learning rate = {}'.format(FLAGS.learning_rate))
print('num epochs = {}'.format(FLAGS.num_epochs))
print('batch size = {} \n'.format(FLAGS.batch_size))


# start training and evaluation
train_val_model(model, device, optimizer, criterion, train_dataloader, vall_dataloader, class_names, FLAGS.num_epochs)

# plot charts:

train_data = pd.read_csv('./outputs/files/train_data.csv')
vall_data = pd.read_csv('./outputs/files/vall_data.csv')

plt_charts(train_data['loss'], train_data['accuracy'], vall_data['loss'], vall_data['accuracy'])


# plot confusion matrix:

data_arr = pd.read_csv('./outputs/files/validation_results.csv')

label_index_arr = np.asarray(data_arr['y_true'],dtype=str)
pred_arr = np.asarray(data_arr['y_pred'],dtype=str)

conf_mat = confusion_matrix(label_index_arr, pred_arr)

fig, ax = plt.subplots(figsize=(10,10))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=True, current_ax=ax)
fig.subplots_adjust(left=0.15, bottom=0.28, right=0.94, top=0.99, wspace=0.2 ,hspace=0.20 )

plt.savefig('./outputs/files/validation_results_cm.png')


# show bunch of wrong predicted images to try enhance the model

df = (pd.read_csv('./outputs/files/validation_results.csv')[lambda x: x['y_true'] != x['y_pred']])
                
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

    
plt.savefig('./outputs/files/wrongly_predicted_images_val_ds.png')