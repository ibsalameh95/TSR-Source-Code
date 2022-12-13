from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
from matplotlib import pyplot as plt
import numpy as np 

def plt_charts(train_loss_list, train_acc_list, vall_loss_list, vall_acc_list): 
        print('****** saving figures *******')
        # plotting the data and save figure
        plt.figure('train', (12,6))
        plt.subplot(1,2,1)
        plt.title('Epoch vs Loss')
        
        epoch = [i + 1 for i in range(len(train_loss_list))]
        
        plt.xlabel('epoch')
        
        plt.plot(epoch,train_loss_list, label = 'Train Loss')
        plt.plot(epoch,vall_loss_list, label = 'Validation Loss')
        plt.legend(loc="best")
        
        plt.subplot(1,2,2)
        plt.title('Epoch vs Accuracy')

        plt.xlabel('epoch')
        plt.plot(epoch, train_acc_list, label='Train Accuracy')
        plt.plot(epoch, vall_acc_list, label='Validation Accuracy')
        
        plt.legend(loc="best")
        plt.show
        plt.savefig('./outputs/files/loss_and_acc_vs_epoch.png')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          current_ax = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = (cm2.astype('float') - np.amin(cm2)) / (np.amax(cm2)-np.amin(cm2))
        #print(cm_normalized)
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        cm_normalized = (cm.astype('float') - np.amin(cm)) / (np.amax(cm)-np.amin(cm))

    ax = current_ax
    if normalize:
        im = ax.imshow(cm2, interpolation='nearest', cmap=cmap)
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if title:
        ax.set_title(title, fontweight="bold")
    # plt.colorbar()
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    # ax.set_xticklabels(classes)
    ax.set_xticklabels(classes, rotation=45, horizontalalignment="right",)
    # ax.set_yticklabels(classes)
    ax.set_yticklabels(classes, rotation=45)
    ax.set_ylim( (len(classes)-0.5, -0.5) )


    fmt = '.2%' if normalize else '.2%'
    thresh = 0.5
    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            
            ax.text(j, i, '{}\n({:.2%})'.format(cm[i, j],cm2[i, j]),
                     horizontalalignment="center",verticalalignment="center",
                     fontsize=10,
                     color="white" if cm_normalized[i, j] > thresh else "black")
    else: 
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            ax.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",verticalalignment="center",
                     fontsize=10, 
                     color="white" if cm_normalized[i, j] > thresh else "black")

    ax.set_ylabel('Truth')
    ax.set_xlabel('Predicted')