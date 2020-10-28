# -*- coding: utf-8 -*-
"""Midterm2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18MvBDJ9z2h-apjDlrDCkPHZMXVhAzaXI
"""

# deps
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set to only print out errors
import tensorflow as tf
from keras.utils.np_utils import to_categorical #encode the categories for Cifar100
import numpy as np
from keras.utils.layer_utils import count_params

# to prevent program from completely consuming gpu memory
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# set the seeds for repeatable results
np.random.seed(0)
tf.random.set_seed(0)
tts_seed = 31415 # train test split seed.
#import tensorflow_datasets as tfds


# download dataset
train, test = tf.keras.datasets.cifar10.load_data()

# preprocessing
train_images, train_labels = train
train_images = train_images/255

test_images, test_labels = test
test_images = test_images/255

# define a mobilenet block
depthwise_conv = tf.keras.layers.DepthwiseConv2D
regular_conv = tf.keras.layers.Conv2D
batchnorm = tf.keras.layers.BatchNormalization
activation = tf.keras.layers.ReLU
average_pooling = tf.keras.layers.AveragePooling2D
flatten = tf.keras.layers.Flatten
dropout = tf.keras.layers.Dropout
dense = tf.keras.layers.Dense

def mobilenet_block(stride_2, out_channels, with_dropout=True):
    return [
        depthwise_conv((3, 3), padding='same',
                       strides=((2, 2) if stride_2 else (1, 1))),
        batchnorm(),
        activation(),
        regular_conv(int(out_channels), (1, 1)),
        batchnorm(),
        activation(),


        # experimenting with dropout levels
        # right now all models overfit without dropout
        *([dropout(0.2)] if with_dropout else [])
    ]

def regular_conv_block(stride_2, out_channels, with_dropout=True):
    return [
        regular_conv(int(out_channels), (3, 3), padding='same',
                    strides=((2, 2) if stride_2 else (1, 1))),
        batchnorm(),
        activation()
    ]

# full mobilenet model; run this with rho=1/7 to work with CIFAR-10 without
# further modification (designed for 224x224 image)
def full_mobilenet(alpha=1, rho=1, use_mobilenet_block=True):
    block = mobilenet_block if use_mobilenet_block else regular_conv_block

    model = tf.keras.Sequential([
        tf.keras.Input((int(rho * 224), int(rho * 224), 3)),

        regular_conv(int(alpha * 32), (3, 3), padding='same', strides=(2, 2)),

        *block(False, int(alpha * 64)),
        *block(True, int(alpha * 128)),
        *block(False, int(alpha * 128)),
        *block(True, int(alpha * 256)),
        *block(False, int(alpha * 256)),
        *block(True, int(alpha * 512)),

        *block(False, int(alpha * 512)),
        *block(False, int(alpha * 512)),
        *block(False, int(alpha * 512)),
        *block(False, int(alpha * 512)),
        *block(False, int(alpha * 512)),

        *block(True, int(alpha * 1024)),

        average_pooling((int(rho * 7), int(rho * 7))),
        flatten(),
        dropout(0.2),

        dense(1000),

        # this differs from ImageNet because of number of classes
        dense(10)
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    return model

# model def: run this for 20 epochs to get ~80% accuracy
# uses larger weights than v2, but ~800k params
def cifar_mobilenet_v1(alpha=1, rho=1, use_mobilenet_block=True):
    block = mobilenet_block if use_mobilenet_block else regular_conv_block

    model = tf.keras.Sequential([
        tf.keras.Input((int(rho * 32), int(rho * 32), 3)),

        regular_conv(int(alpha * 32), (3, 3), padding='same'),

        *block(False, alpha * 64),
        *block(False, alpha * 128),
        *block(True, alpha * 256),
        *block(True, alpha * 512),
        *block(True, alpha * 1024),

        average_pooling((int(rho * 4), int(rho * 4))),
        flatten(),
        dropout(0.2),
        dense(10)
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    return model

# model def: alternate version; run this with 50 epochs to get ~80% accuracy
# uses smaller weights than v1, only ~50k params, only increase weights when 
# reducing image dimensions like in original mobilenet
def cifar_mobilenet_v2(alpha=1, rho=1, use_mobilenet_block=True):
    block = mobilenet_block if use_mobilenet_block else regular_conv_block

    model = tf.keras.Sequential([
        tf.keras.Input((int(rho * 32), int(rho * 32), 3)),

        regular_conv(int(alpha * 32), (3, 3), padding='same'),

        *block(False, alpha * 32),
        *block(False, alpha * 32),
        *block(True, alpha * 64),
        *block(True, alpha * 128),
        *block(True, alpha * 256),

        average_pooling((int(rho * 4), int(rho * 4))),
        flatten(),
        dropout(0.2),
        dense(10)
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

    return model

# run model, collect summary and history
def run_model(model_class, num_epochs, run_model=True, model_parms={}):
    # create_model
    model = model_class(**model_parms)

    metadata = {
        'model': 'v1' if model_class == cifar_mobilenet_v1 else 'v2',
        'epochs': num_epochs,
        'params': model_parms,
        'summary': '',
        'num_trainable_params': count_params(model.trainable_weights)
    }

    # save model summary to a string (not the default)
    def save_model_summary_to_string(summary_line):
        nonlocal metadata
        metadata['summary'] = metadata['summary'] + summary_line + '\n'
    model.summary(print_fn=save_model_summary_to_string)

    # run model
    if run_model:
        # don't resize feature sets
        resized_train_images = train_images
        resized_test_images = test_images

        # resize images based on rho; note this also depends on the model's
        # default image size (32 for our model, 224 for original model)
        rho = model_parms['rho'] if 'rho' in model_parms else 1
        if rho != 1:
            default_size = 224 if model_class == full_mobilenet else 32
            resized_train_images = tf.image.resize(
                resized_train_images,
                (int(rho * default_size), int(rho * default_size))
            )
            resized_test_images = tf.image.resize(
                resized_test_images,
                (int(rho * default_size), int(rho * default_size))
            )

        # train
        history = model.fit(
            resized_train_images, train_labels,
            epochs=num_epochs, validation_split=0.2, 
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=10, min_delta=0.005)]).history

        # final test accuracy
        test_accuracy = model.evaluate(resized_test_images, test_labels, verbose=1, return_dict=True)
    else:
        history = None
        test_accuracy = None

    return {
        'metadata': metadata,
        'history': history,
        'test_accuracy': test_accuracy
    }

print(run_model(full_mobilenet, 0, run_model=False)['metadata']['summary'])
exit(0)

models = [cifar_mobilenet_v1, cifar_mobilenet_v2]
epochs = [100]
alphas = [2, 1, 7/8, 0.75, 0.5]
rhos = [1, 7/8, 0.75, 0.5]

results = []

grid_count = len(models) * len(epochs) * len(alphas) * len(rhos)
i = 1

for model in models:
    for epoch in epochs:
        for alpha in alphas:
            for rho in rhos:
                print(f'iteration {i}/{grid_count}: alpha: {alpha}; rho: {rho}; epochs: {epoch}; model: {model}')
                i += 1
                results.append(run_model(model, epoch, model_parms={
                    'alpha': alpha,
                    'rho': rho,
                    # 'use_mobilenet_block': False,
                    # 'run_model': False
                }))

import pickle
from datetime import datetime

pickle.dump(results, open('mobilev1v2-full-cifar10-' + datetime.now().strftime("%y-%m-%d-%H%M") + '.pkl', 'wb'))

