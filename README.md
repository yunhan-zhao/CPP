# CPP
This repo contains the Pytorch implementation of:

[Camera Pose Matters: Improving Depth Prediction by Mitigating Pose Distribution Bias](https://openaccess.thecvf.com/content/CVPR2021/html/Zhao_Camera_Pose_Matters_Improving_Depth_Prediction_by_Mitigating_Pose_Distribution_CVPR_2021_paper.html)

[Yunhan Zhao](https://www.ics.uci.edu/~yunhaz5/), [Shu Kong](http://www.cs.cmu.edu/~shuk/), and [Charless Fowlkes](https://www.ics.uci.edu/~fowlkes/)

CVPR 2021 (oral)

For more details, please check our [project website](https://www.ics.uci.edu/~yunhaz5/cvpr2021/cpp.html)

### Abstract
Monocular depth predictors are typically trained on large-scale training sets which are naturally biased w.r.t the distribution of camera poses. As a result, trained predictors fail to make reliable depth predictions for testing examples captured under uncommon camera poses. To address this issue, we propose two novel techniques that exploit the camera pose during training and prediction. First, we introduce a simple perspective-aware data augmentation that synthesizes new training examples with more diverse views by perturbing the existing ones in a geometrically consistent manner. Second, we propose a conditional model that exploits the per-image camera pose as prior knowledge by encoding it as a part of the input. We show that jointly applying the two methods improves depth prediction on images captured under uncommon and even never-before-seen camera poses. We show that our methods improve performance when applied to a range of different predictor architectures. Lastly, we show that explicitly encoding the camera pose distribution improves the generalization performance of a synthetically trained depth predictor when evaluated on real images.

## Reference
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

## Note
The core part of the code is released now! The detailed training, evaluation, dataset scripts, and pretrained models will be released in the next few weeks.

## Contents

- [Requirments](#requirements)
- [Dataset]
- [Training]
- [Evaluation]
- [Pretrained Models]


## Requirements
1. Python 3.6 with Ubuntu 16.04
2. Pytorch 1.1.0
3. Apex 0.1 (optional)

You also need other third-party libraries, such as numpy, pillow, torchvision, and tensorboardX (optional) to run the code. We use apex when training all models but it is not strictly required to run the code. 

