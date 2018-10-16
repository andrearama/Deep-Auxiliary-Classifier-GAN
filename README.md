# Deep-Auxiliary-Classifier-GAN
High quality image generation with a modified version of Auxiliary Classifier GAN

The work is an expansion of the AC-GAN architecture introduced in the paper "Conditional Image Synthesis With Auxiliary Classifier GANs" (https://arxiv.org/pdf/1610.09585.pdf).
While in this paper the images are generated are 64x64 or 128x128, the aim of this implementation is to generate bigger images, namely 300x300.

## Getting Started
### Installation
- The requred libraries are opencv, numpy and keras (any recent versions would be fine).
You can install all the dependencies by:
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/andrearama/Deep-Auxiliary-Classifier-GAN
cd pytorch-CycleGcd Deep-Auxiliary-Classifier-GAN
```

### Train the model
Just run the train script. It is possible to change the default standards (look at the main file for more information)
```bash
python train.py 
```
### Sources:
https://github.com/andrearama/Keras-GAN
https://github.com/soumith/ganhacks
