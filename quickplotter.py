import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from datetime import datetime

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set to only print out errors
#import tensorflow as tf
#from keras.utils.np_utils import to_categorical #encode the categories for Cifar100

## to prevent program from completely consuming gpu memory
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

# set the seeds for repeatable results
np.random.seed(0)
#tf.random.set_seed(0)
tts_seed = 31415 # train test split seed.

#set fonts
font = {'family': 'cmr10', 'size': 18}
mpl.rc('font', **font)  # change the default font to Computer Modern Roman
mpl.rcParams['axes.unicode_minus'] = False  # because cmr10 does not have a Unicode minus sign

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

def plt_func(num_classes, num_epochs, accuracy, val_accuracy, loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(figsize=(8.5, 11), nrows=2, ncols=1)
    ax1.plot(range(1,num_epochs + 1), accuracy, label='Training Accuracy')
    ax1.plot(range(1,num_epochs + 1), val_accuracy, label='Validation Accuracy')
    ax1.set_title('Cifar' + str(num_classes) + ' Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Ratio', labelpad=25).set_rotation(0)
    ax1.legend()

    ax2.plot(range(1,num_epochs + 1), loss, label='Training Loss')
    ax2.plot(range(1,num_epochs + 1), val_loss, label='Validation Loss')
    ax2.set_title('Cifar' + str(num_classes) + ' Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss', labelpad=25).set_rotation(0)
    ax2.legend()

    fig.suptitle('This is a somewhat long figure title', fontsize=16)
    fig.tight_layout()
    #fig.savefig('hw4-cifar' + str(num_classes) + '-accuracy-loss.pdf', format='pdf')

params = pickle.load(open('mobilev2-full-cifar10-20-10-26-1805.pkl', 'rb'))

#print(params[0]['metadata'])


for i in range(len(params)):
    print(params[i]['metadata']['num_trainable_params'])
    plt_func(10, params[i]['metadata']['epochs'], params[i]['history']['accuracy'], params[i]['history']['val_accuracy'], params[i]['history']['loss'], params[i]['history']['val_loss'])

#x = []
#y = []

#for i in range(len(params)):
#    x.append(params[i]['metadata']['num_trainable_params'])
#    y.append(params[i]['test_accuracy'])

#fig, ax = plt.subplots()
#ax.scatter(x, y)
#ax.set_xscale('log')
##ax.set_ylim(bottom=0) 

#for i, txt in enumerate(range(len(params))):
#    ax.annotate(txt, (x[i], y[i]))
plt.show()

