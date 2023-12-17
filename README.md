# This is the repository containing the implementation of the paper: 
## Continuous Authentication Using Self-Supervised Learning and Gesture Patterns
### 1) Requirements:
* Python >= 2.9
* Pytorch >=1.13
* ptflops >= 0.6.9
  
### 1) Pre-training:
 ```
 python main.py --net GestureNet --dataset HMOG_ID --train_phase True --input_planes 9 --epoch 1000
 ```
### 2) User verification:

 ```
 python test.py --net GestureNet --dataset HMOG_VER --input_planes 9 --epoch 500 --num_classes = 2
 ```
### 3) User identification:
 ```
 python test.py --net GestureNet --dataset HMOG_ID --train_phase True --input_planes 9 --epoch 500
 ```
### 4) Computational complexity:
 
 ```
 python complexity.py --net GestureNet --repetitions 500
 ```
### 5) Analysis of standard deviation of gaussian noise:

```
python roc_curves_std
```
### 6) Fine-tuning on small annotated dataset:
```
null
```

