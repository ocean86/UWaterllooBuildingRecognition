import time
import os
from PIL import Image
import numpy as np
import random
import tflearn
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
import pdb


# Parameter definitions
#image_size = 600 * (400 - 20)
image_size = 160 * 100
labels_size = 3
labelDict = {'mc':0, 'm3':1, 'dc':2}
# Change below
batch_size = 200
max_steps = 20
hidden_layers = [100, 100]

def getTrainingData():
    images = []
    labels = []
    items = os.listdir('./trainingSet')
    random.shuffle(items)
    for i in items:
        label = labelDict[os.path.basename(i)[:2]]
        #file = tf.read_file(os.path.join('./trainingSet', i))
        #img = tf.image.decode_jpeg(file, channels=3)
        #data = tf.image.convert_image_dtype(img, tf.float32)
        img = Image.open(os.path.join('./trainingSet', i))
        img.load()
        img = np.asarray(img, dtype="float32")
        #img.flatten()
        #img = img.reshape(image_size, 3)
        labels.append(label)
        images.append(img)

    images = np.asarray(images)
    labels = np.asarray(labels)
    labels = to_categorical(labels, 3)
    return images, labels


def getTestingData():
    images = []
    labels = []
    items = os.listdir('./testSet')
    random.shuffle(items)
    for i in items:
        label = labelDict[os.path.basename(i)[:2]]
        #file = tf.read_file(os.path.join('./trainingSet', i))
        #img = tf.image.decode_jpeg(file, channels=3)
        #data = tf.image.convert_image_dtype(img, tf.float32)
        img = Image.open(os.path.join('./testSet', i))
        img.load()
        img = np.asarray(img, dtype="float32")
        #img.flatten()
        #img = img.reshape(image_size, 3)
        labels.append(label)
        images.append(img)

    images = np.asarray(images)
    labels = np.asarray(labels)
    labels = to_categorical(labels, 3)
    return images, labels

def getFittingData(training_data, training_labels):
    images = []
    labels = []
    for i in np.random.choice(len(training_data), int(len(training_data) * 0.2)):
        images.append(training_data[i])
        labels.append(training_labels[i])
    # print(len(images), print(len(labels)))
    return np.asarray(images), np.asarray(labels)


def main():
    beginTime = time.time()
    # Get data
    training_data, training_labels = getTrainingData()
    testing_data, testing_labels = getTestingData()
    fitting_data, fitting_labels = getFittingData(training_data, training_labels)
    print("Data Prepared.")

    # Define Networks
    network = input_data(shape=[None, 100, 160, 3])

    # 1: Convolution layer with 32 filters, each 3x3x3
    conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

    # 2: Max pooling layer
    network = max_pool_2d(conv_1, 2)

    # 3: Convolution layer with 64 filters
    conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

    # 4: Convolution layer with 64 filters
    conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

    # 5: Max pooling layer
    network = max_pool_2d(conv_3, 2)

    # 6: Fully-connected 256 node layer
    network = fully_connected(network, 256, activation='relu')

    # 8: Fully-connected layer with three outputs
    network = fully_connected(network, 3, activation='softmax')

    # Configure how the network will be trained
    acc = Accuracy(name="Accuracy")
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.0005, metric=acc)

    # Wrap the network in a model object
    model = tflearn.DNN(network, checkpoint_path='UWMathBuilding_model/UWBuildingCheckPoint.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='UWMathBuilding_model/tflearn_logs/')

# Run
    model.fit(training_data, training_labels, validation_set=(testing_data, testing_labels), batch_size=200,
        n_epoch=5, run_id='test1', show_metric=True)

    endTime = time.time()
    print('Total time: {:5.2f}s'.format(endTime - beginTime))

if __name__ == '__main__':
    main()
