import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
import struct

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set to only print out errors
import tensorflow as tf

#tf.config.set_visible_devices([], 'GPU')  # use the CPU instead of GPU

# set the seeds for repeatable results
np.random.seed(0)
tf.random.set_seed(0)
tts_seed = 0 # train test split seed.

#set fonts
font = {'family': 'cmr10', 'size': 18}
mpl.rc('font', **font)  # change the default font to Computer Modern Roman
mpl.rcParams['axes.unicode_minus'] = False  # because cmr10 does not have a Unicode minus sign

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

def open_mnist_images_and_labels(image_file, label_file):
    #expect 4 and 2 32 bit (4 byte) big endian integers for headers of imgs and lbls
    with open(image_file, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        # check the magic number
        if magic == 2051:
            images = np.fromfile(f, dtype=np.uint8).reshape(n, rows*cols)
        else:
            images = None
        with open(label_file, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            if magic == 2049:
                labels = np.fromfile(f, dtype=np.uint8)
            else:
                labels = None
        # print(imgs[1].reshape(rows, cols))
        # print(labels)
        return images.reshape(-1, 28, 28, 1), labels

# a nifty little helper function to print numbers
def image_printer(img, rows=28, cols=28):
    img = img.reshape(rows, cols)
    for row in img:
        for col in row:
            col = 'o' if col > 0 else '.'
            print(col, end=' ')
        print()

def main():
    # Hyperparameters: goal was to get 95.5% accuracy on both training and validation in hopes that testing accuracy
    # would reflect that

    # portion of the training data used for validation
    valid_ratio = 0.2
    # I tried 25 epochs initially, but 20 seemed to achieve the desired training and validation accuracy
    num_epochs = 3
    # I left the batch size at the default 32 for the longest time and was getting good results, but I tried 64 and 128
    # and got slightly higher training and validation accuracies
    batch_size = 128
    # originally tried learning rate = 0.001 but that was rather unstable so I halved it to get a smoother curve
    learning_rate = 0.0005
    # I started with a dropout rate of 0.5 (as described in class) and arrived at this value
    dropout_rate = 0.15
    # finally I tried changing the train test split seed during training and validation not to tweak the results, but
    # just to ensure that I wasn't getting a bad selection of data. Less necessary with this than say for frequentist
    # ML, but was a good sanity check nonetheless.

    train_images, train_labels = open_mnist_images_and_labels('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
    image_printer(train_images[0])
    trainX, validX, trainY, validY = train_test_split(train_images, train_labels,
                                                      test_size=valid_ratio, random_state=tts_seed)
    print('Sanity check of shapes:', trainX.shape, validX.shape, trainY.shape, validY.shape, sep='\n')

    # initially tried two fully connected layers w/128 neurons each but then decided to try 64 neurons, and that seemed
    # to still achieve a good training and validation accuracy (both above 95.5 accuracy) so I used that instead.
    # I had a dropout and l2 regularization at each layer, and the dropout rate selection is described above.

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.DepthwiseConv2D((3,3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(256, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(256, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu', strides=2, kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(512, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),

        tf.keras.layers.DepthwiseConv2D((3, 3), kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(512, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.DepthwiseConv2D((3, 3), kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(512, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.DepthwiseConv2D((3, 3), kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(512, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.DepthwiseConv2D((3, 3), kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(512, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.DepthwiseConv2D((3, 3), kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(512, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),

        tf.keras.layers.DepthwiseConv2D((3, 3), strides=2, kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(1024, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.DepthwiseConv2D((3, 3), kernel_initializer='he_uniform', activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1024, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.AveragePooling2D((7,7)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    print(model.summary())

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])

    history = model.fit(trainX, trainY, epochs=num_epochs, validation_data=(validX, validY), batch_size=batch_size)

    fig, (ax1, ax2) = plt.subplots(figsize=(8.5, 11), nrows=2, ncols=1)
    ax1.plot(range(1,num_epochs + 1), history.history['accuracy'], label='Training Accuracy')
    ax1.plot(range(1,num_epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Ratio', labelpad=25).set_rotation(0)
    ax1.legend()

    ax2.plot(range(1,num_epochs + 1), history.history['loss'], label='Training Loss')
    ax2.plot(range(1,num_epochs + 1), history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss', labelpad=25).set_rotation(0)
    ax2.legend()

    fig.tight_layout()
    fig.show() #fig.savefig('hw3-accuracy-loss.pdf', format='pdf')

    # I used this very janky piece of code to prevent prematurely running the model on the test set I could have saved
    # the model and loaded it, but decided against that only after tweaking the hyperparameters until I was satisfied
    # with the training and validation accuracy did I change this to False (I confess I did run the test set two to
    # three times to try some crazy models but I didn't end up using those results).
    if False:
        return

    test_images, test_labels = open_mnist_images_and_labels('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')

    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    num_correct = sum(1*(predicted_labels == test_labels))
    print('Accuracy on Test Set:', num_correct/len(test_labels) * 100.0)

main()
