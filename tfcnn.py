import tensorflow as tf
import time
import os
from PIL import Image
import numpy as np
import random
import pdb

tf.logging.set_verbosity(tf.logging.ERROR)

# Parameter definitions
image_size = 600 * (400 - 20)
labels_size = 3
labelDict = {'dc':0, 'mc':1, 'm3':2}
# Change below
batch_size = 300
max_steps = 50
hidden_layers = [128, 128]

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
        img = np.asarray(img, dtype="float32").flatten()
        #img = img.reshape(image_size, 3)
        labels.append(label)
        images.append(img)

    images = np.asarray(images)
    labels = np.asarray(labels)
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
        img = np.asarray(img, dtype="float32").flatten()
        #img = img.reshape(image_size, 3)
        labels.append(label)
        images.append(img)

    images = np.asarray(images)
    labels = np.asarray(labels)
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

    # Define the Estimator
    feature_columns = [tf.feature_column.numeric_column("", shape=[684000])]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=hidden_layers,
                                              n_classes=labels_size,
                                              model_dir="./UWMathBuilding_model")

    # Train model.
    classifier.fit(x=training_data, y=training_labels, batch_size=batch_size, steps=max_steps)
    endTime = time.time()
    print('Training Finished. Time: {:5.2f}s'.format(endTime - beginTime))

    # Evaluate
    # On Training imgaes:
    test_accuracy = classifier.evaluate(x=fitting_data, y=fitting_labels, steps=1)["accuracy"]
    print("\nExample accuracy: %g %%" % (test_accuracy * 100))
    predictions = list(classifier.predict(x=fitting_data))
    labels = list(fitting_labels)
    print("Prediction:" + str(predictions))
    print("True Value:" + str(labels))
    # On Testing images
    test_accuracy = classifier.evaluate(x=testing_data, y=testing_labels, steps=1)["accuracy"]
    print("\nTesting Data accuracy: %g %%" % (test_accuracy * 100))
    #predictions = list(classifier.predict(x=testing_data))
    #labels = list(testing_labels)
    #print("Prediction:" + str(predictions))
    #print("True Value:" + str(labels))

    endTime = time.time()
    print('Total time: {:5.2f}s'.format(endTime - beginTime))

if __name__ == '__main__':
    main()
