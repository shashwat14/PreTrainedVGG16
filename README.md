# PreTrainedVGG16
This a pre trained VGG16 model that is meant for usage in predicting, finetuning or entirely new projects. This VGG16 network also provides extensive visualization during training for the purpose of debugging (and) or hyperparamter seach. Example usage scenarios:
1. Classify image in 1 among 1000 categories of ImageNet - Not that useful
2. Finetune VGG16 for classifying images from datasets apart from ImageNet - Useful
3. Using the the convolutoinal codes (7*7*512 dimension) vector and appending your custom layers - Very useful

## Installation
I have borrowed the pre-trained weights from the following link - https://www.cs.toronto.edu/~frossard/post/vgg16/
1. Git clone this project. Let's call the directory $root
2. Download the weights file (vgg16_weights.npz) and place in $root

## Classifying images in 1 among 100 categories form ImageNet
To be finished
