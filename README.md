# Code of my Master Thesis
## Topic: End-to-end Machine Learning Systems for Visual Localization

## Abstract
In this work, we present ViPNet, a robust and real-time capable monocular camera pose regression network. We train a convolutional neural network to estimate the six degrees of freedom camera pose from a single monocular image in an end-to-end manner. In order to estimate camera poses with uncertainty, we use a Bayesian version of the ResNet-50 as our basic network. SE-Blocks are applied in residual units to increase our model's sensitivity to informative features. Our ViPNet is trained using a geometric loss function with trainable parameters, which can simplify the fine tuning process significantly. We evaluate our ViPNet on the Cambridge Landmarks dataset and also on our Karl-Wilhelm-Plaza dataset, which is recorded with a experiment vehicle. As evaluation results, our ViPNet outperforms other end-to-end monocular camera pose estimation methods. Besides the high accuracy, our ViPNet requires only 9-15ms to predict one camera pose, which allows us to estimate camera poses with a very high frequency.

# Requirements:
```python
torch == 1.4.0
torchvision == 0.5.0
tensorboardX == 2.0
scikit-image == 0.16.2
Pillow == 6.2.1
numpy == 1.17.4
pandas == 0.25.3
```
