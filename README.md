# UWaterlooBuildingRecognition
A machine learning project to recognize UWaterloo buildings from pictures. 
Training Images are generated from Google Maps StreetView API. Samples are upload in dataset folder.

## Steps:

#### 0. Collect Training Data
```bash
python dataset/downloadImages.py
```
This script collects photos around a building via Google Maps StreetView API. 
Set the location(center) of the building, start and end camera location, number of photos wanted at the top of the file. And run this script to download images.

#### 1. create training & testing dataset by: 
```bash
python splitData.py
```
The default ratio of Test:Train is 1:9. You can change it inside the python file.
This sciptes crops out the google logo under the image and resize it to be smaller. 
New images saved under trainnningSet/ and testSet/.

#### 2. Training and Testing: learnCnn.py
```bash
python learnCnn.py
```
The network consits of: 3 Conv-Layer, 2 Max-pooling Layer, 2 Fully Connected Layer.
