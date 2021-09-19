# Camera Pose Matters: Improving Depth Prediction by Mitigating Pose Distribution Bias

This repo contains the official Pytorch implementation of:

[Camera Pose Matters: Improving Depth Prediction by Mitigating Pose Distribution Bias](https://openaccess.thecvf.com/content/CVPR2021/html/Zhao_Camera_Pose_Matters_Improving_Depth_Prediction_by_Mitigating_Pose_Distribution_CVPR_2021_paper.html)

[Yunhan Zhao](https://www.ics.uci.edu/~yunhaz5/), [Shu Kong](http://www.cs.cmu.edu/~shuk/), and [Charless Fowlkes](https://www.ics.uci.edu/~fowlkes/)

CVPR 2021 (oral)

For more details, please check our [project website](https://www.ics.uci.edu/~yunhaz5/cvpr2021/cpp.html)

### Abstract
Monocular depth predictors are typically trained on large-scale training sets which are naturally biased w.r.t the distribution of camera poses. As a result, trained predictors fail to make reliable depth predictions for testing examples captured under uncommon camera poses. To address this issue, we propose two novel techniques that exploit the camera pose during training and prediction. First, we introduce a simple perspective-aware data augmentation that synthesizes new training examples with more diverse views by perturbing the existing ones in a geometrically consistent manner. Second, we propose a conditional model that exploits the per-image camera pose as prior knowledge by encoding it as a part of the input. We show that jointly applying the two methods improves depth prediction on images captured under uncommon and even never-before-seen camera poses. We show that our methods improve performance when applied to a range of different predictor architectures. Lastly, we show that explicitly encoding the camera pose distribution improves the generalization performance of a synthetically trained depth predictor when evaluated on real images.

### Reference
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{zhao2021camera,
  title={Camera Pose Matters: Improving Depth Prediction by Mitigating Pose Distribution Bias},
  author={Zhao, Yunhan and Kong, Shu and Fowlkes, Charless},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15759--15768},
  year={2021}
}
```

### Contents
- [Requirments](#requirements)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)


### Requirements
1. Python 3.6 with Ubuntu 16.04
2. Pytorch 1.1.0
3. Apex 0.1 (optional)

You also need other third-party libraries, such as numpy, pillow, torchvision, and tensorboardX (optional) to run the code. We use apex when training all models but it is not strictly required to run the code. 

### Dataset
We use InteriorNet and ScanNet in this project. The detailed data file lists are located in `dataset` folder where each file correspinds to one training/testing distribution (natural, uniform or restricted). Please download and extract the appropriate files before training.
####  Dataset Structure (e.g. interiorNet_training_natural_10800)
```
interiorNet_training_natural_10800
    | rgb
        | rgb0.png
        | ...
    | depth
        | depth0.png
        | ...
    cam_parameter.txt
```
`cam_parameter.txt` contains the intrinsics and camera pose for each sample in the subset. Feel free to sample your own distribution and train with your own data. 

### Training
All training steps use one common `train.py` file so please make sure to comment/uncomment for training with CPP, PDA, or CPP + PDA.
```bash
CUDA_VISIBLE_DEVICES=<GPU IDs> python train.py \
  --data_root=<your absolute path to InteriorNet or ScanNet> \
  --training_set_name=interiorNet_training_natural_10800 \
  --testing_set_name=interiorNet_testing_natural_1080 \
  --batch_size=12 --total_epoch_num=200 --is_train --eval_batch_size=10
```
`batch_size` and `eval_batch_size` are flexible to change given your working environment. Feel free to swap `interiorNet_training_natural_10800` and `interiorNet_testing_natural_1080` to train and test on different distributions.

### Evaluations
Evaluate the final results
```bash
CUDA_VISIBLE_DEVICES=<GPU IDs> python train.py \
  --data_root=<your absolute path to InteriorNet or ScanNet> \
  --training_set_name=interiorNet_training_natural_10800 \
  --testing_set_name=interiorNet_testing_natural_1080 \
  --eval_batch_size=10
``` 
If you want to evaluate with your own data, please create your own testing set with the dataset structure described above.

### Pretrained Models
Pretrained models will be uploaded soon.

### Questions
Please feel free to email me at (yunhaz5 [at] ics [dot] uci [dot] edu) if you have any questions.