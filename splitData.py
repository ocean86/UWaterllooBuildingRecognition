# Generate Train and Test data from esisting images
import random
import os
import shutil
from PIL import Image
import pdb


# crop Google Logo and save to new Destination
def cropLogoAndSave(source, destinaiton):
    img = Image.open(i)
    width = img.size[0]
    height = img.size[1]
    imgCrop = img.crop((0, 0, width, height - 20)) # Logo height is approx 20
    imgCrop = imgCrop.resize((160, 100))
    imgCrop.save(destinaiton, "JPEG")


# set Ratio of Test images 
spiltRatio = 1 / 10 # Test : Total

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
    random.shuffle(images)
    numberOfTestData = int(len(images) * spiltRatio)
    testData = random.sample(images, numberOfTestData)
    testCount = 1
    trainCount = 1
    for i in images:
        if i in testData:
            dstName = os.path.basename(b) + '_' + str(testCount) + '.jpg'
            cropLogoAndSave(i, os.path.join('./testSet', dstName))
            # shutil.copy(i, os.path.join('./testSet', dstName))
            testCount += 1
        else:
            dstName = os.path.basename(b) + '_' + str(trainCount) + '.jpg'
            cropLogoAndSave(i, os.path.join('./trainingSet', dstName))
            # shutil.copy(i, os.path.join('./trainingSet', dstName))
            trainCount += 1
    print(os.path.basename(b), testCount, trainCount)
