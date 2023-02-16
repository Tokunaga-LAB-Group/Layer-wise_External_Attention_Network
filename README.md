# LEA-Net : Layer-wise_External_Attention_Network

<!-- Badges -->
<p>
    <img src="https://img.shields.io/badge/LEA_Net-1.0-FFF.svg?style=flat">
    <img src="https://img.shields.io/badge/Python-3.8.12-3776AB.svg?style=flat&logo=Python">
    <img src="https://img.shields.io/badge/TensorFlow-2.4.1-FF6F00.svg?style=flat&logo=TensorFlow">
    <img src="https://img.shields.io/badge/TF_addons-0.14.0-FF6F00.svg?style=flat">
    <img src="https://img.shields.io/badge/Keras-2.3.1-D00000.svg?style=flat&logo=Keras">
    <img src="https://img.shields.io/badge/NumPy-1.19.5-013243.svg?style=flat&logo=NumPy">
    <img src="https://img.shields.io/badge/OpenCV-4.5.3.56-5C3EE8.svg?style=flat&logo=OpenCV">
    <img src="https://img.shields.io/badge/pandas-1.3.3-150458.svg?style=flat&logo=pandas">
    <img src="https://img.shields.io/badge/Pillow-8.4.0-000000.svg?style=flat">
    <img src="https://img.shields.io/badge/scikit_image-0.18.3-F7931E.svg?style=flat">
    <img src="https://img.shields.io/badge/scikit_learn-0.24.2-F7931E.svg?style=flat&logo=scikit-learn">
</p>

## Overview
<div align="center">
    <img src="README_Figures/Overview_LEA-Net.png">
</div>
This page provides the implementation of LEA-Net (Layer-wise External Attention Network). The formative anomalous regions on the intermediate feature maps can be highlighted through layer-wise external attention. LEA-Net has a role in boosting existing CNN anomaly detection performances.

## Usage
### phase 1: Unsupervised Learning.
In this phase, images are reconstructed by various methods.
#### Setting up Dataset
Make sure that it follows the following data tree:
```
Dataset___Positive___image
        |          |_image
        |          |_Directory___image
        |               :      |_image
        |                          :
        |_Negative___image
                   :
```
train.py loads all images in the Positive/Negative directory.

#### Training Reconstruction model to make Anomaly Attention Map (AAN)
For example, you can run this sample code:
```
python phase1.py \
--GPU ${GPU_ID} --save_path ${SAVE_PATH} \
--dataset ${DATASET}  --task "colorization" --n_splits 10 \
--model "Unet" --loss "BCE" \
--batch_size 16 --epochs 10 \
--lr 1e-04
```


### phase 2: Supervised Learning.
In this phase, images are reconstructed by various methods.
#### Setting up Dataset
Please specify the output of phase 1. Make sure that it follows the following data tree:
```
Dataset___01___GT___Train___image1
        |    |    |       |_image2
        |    |    |           :
        |    |    |
        |    |    |_Test___image1
        |    |           |_image2
        |    |               :
        |    |
        |    |_Anomap___Train___image1
        |    |        |       |_image2
        |    |        |           :
        |    |        |
        |    |        |_Test___image1
        |    |               |_image2
        |    |                   :
        |    |
        |    |_Label.npz
        |
        |_02___...
        :
        |_'n_splits (phase1)'___...
```
#### Training Classifier model
For example, you can run this sample code:
```
python phase2.py \
--GPU  ${gpu_id} --save_path ${SAVE_PATH} \
--dataset ${DATASET} --n_splits 10 \
--input_method 'img_and_attMap' \
--output_method 'AAN_and_ADN' \
--model_ADN 'ResNet18' --model_AAN 'MobileNet' \
--attention_points 0 \
--batch_size 16 --epochs 100 --lr 1e-04 \
--epochs 3
```

## Performance
#### Effects of layer-wise external attention
Visualization of feature maps at attention points before and after the layer-wise external attention.

<div align="center">
    <img src="README_Figures/Overview_LEA-Net.png">
</div>

The data in the figure can be downloaded from [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) and [DR2](https://figshare.com/articles/dataset/Advancing_Bag_of_Visual_Words_Representations_for_Lesion_Classification_in_Retinal_Images/953671/3).