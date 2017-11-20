# Generate Train and Test data from esisting images
import random
import os
import shutil

# split Methods can  'half'
spiltRatio = 3 / 10 # Test : Total

buildings = [os.path.join('./dataset', d) for d in os.listdir('./dataset') if os.path.isdir(os.path.join('./dataset', d))]

# Empty folders
if os.path.exists('./trainingSet'):
    shutil.rmtree('./trainingSet/')
os.makedirs('./trainingSet')
if os.path.exists('./testSet'):
    shutil.rmtree('./testSet/')
os.makedirs('./testSet')


for b in buildings:
    images = [os.path.join(b, i) for i in os.listdir(b) if i.endswith('jpg')]
    numberOfTestData = int(len(images) * (3 / 10))
    testData = random.sample(images, numberOfTestData)
    testCount = 1
    trainCount = 1
    for i in images:
        if i in testData:
            dstName = os.path.basename(b) + '_' + str(testCount) + '.jpg'
            shutil.copy(i, os.path.join('./testSet', dstName))
            testCount += 1
        else:
            dstName = os.path.basename(b) + '_' + str(trainCount) + '.jpg'
            shutil.copy(i, os.path.join('./trainingSet', dstName))
            trainCount += 1
    print(os.path.basename(b), testCount, trainCount)
