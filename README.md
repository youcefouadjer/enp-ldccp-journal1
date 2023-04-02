# This is the repository containing the implementation of the paper: 
## Towards Self-supervised Learning of Human Identity from Gesture Patterns

### 1) Requirements:
* Python >= 2.9
* Pytorch >=1.13
* ptflops >= 0.6.9

### 2) Self-supervised pre-training:

* The following command is used for contrastive pre-training

 ```
 python main.py --net GestureNet --dataset HMOG_ID --train_phase True --input_planes 9 --epoch 1000
 
 ```
 
 ### 3) Downstream evaluation:
 
 ```
 python test.py --net GestureNet --dataset HMOG_ID --input_planes 9 --epoch 500 --num_classes = 2
 ```
 
 
