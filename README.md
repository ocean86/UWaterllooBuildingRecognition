# UWaterlooBuildingRecognition
A machine learning project to recognize UWaterloo buildings from pictures. 
Training Images are generated from Google Maps StreetView API. Samples are upload in dataset folder.

## Steps:
### 0. Collect Training Data

### 1. cd to the directory of this project
***dataset/downloadImages.py*** collects photos around a building via Google Maps StreetView API. 
Set the location(center) of the building, start and end camera location, number of photos wanted at the top of the file. And run this script to download images.

### 2. create training & testing dataset by: 
```bash
python splitData.py
```
The default ratio of Test:Train is 1:9. You can change it inside the python file.

### 3. Training and Testing: learnCnn.py
```bash
python learnCnn.py
```
The network consits of: 3 Conv-Layer, 2 Max-pooling Layer, 2 Fully Connected Layer.
