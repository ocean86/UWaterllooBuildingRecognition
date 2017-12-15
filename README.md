# UWaterlooBuildingRecognition
A machine learning project to recognize UWaterloo buildings from pictures. 

## To Run:

### 1. cd to the directory of this project

### 2. create training & testing dataset by: 
```bash
python spiltData.py
```
The default ratio of Test:Train is 1:9. You can change it inside the python file.

### 3. Set parameters in tfcnn.py:
```python
batch_size = 10
max_steps = 100
hidden_layers = [64, 32]
```
**batch_size** 是一次跑多少图。 因为太多图的话内存可能装不下, 会慢。 这个数越大越好,最大为所有training_data. 10 就是train一次从training_data里随机抽10张图来train。

**max_steps** 是train多少次. steps为100的话就是train 100次. 要看batch_size, 如果batch够大的话理论上不需要很多次train. train得越多未必越准确.

**hidden_layers** 表示有多少个hidden layers. 每个hidden layers有多少个node. [64, 32] 表示这个NN有两个layers, 第一个layer有64个nodes, 第二个32个nodes.
理论上单层nodes越多越好.但我试着run了个单层的1024个node, [1024], 要很长时间，硬盘都满了。

