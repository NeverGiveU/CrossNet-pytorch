## Introduction
&ensp;&ensp;&ensp;&ensp;This is an pytorch implementation for paper [CrossNet: Latent Cross-Consistency for Unpaired Image Translation](https://arxiv.org/abs/1901.04530)

&ensp;&ensp;&ensp;&ensp;The project are build based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Thanks a lot to them, and we really got help from their works.
### Tips
For easier understanding, we modify the project such that
#### 1) Simple Project Structure
&ensp;&ensp;&ensp;&ensp;We use a more single and direct structure then the initial project of [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), which can be shown as follows.
```
├─checkpoints
│  └─CrossNet_horse2zebra
├─datasets
├─results
│  └─CrossNet_horse2zebra
├─config.py
├─data.py
├─loss.py
├─networks.py
├─test.py
├─train.py
├─README.md
└─__pycache__
```
#### 2) Convenient Usage
&ensp;&ensp;&ensp;&ensp;Inspired by the initial project management mechanism, we provide an efficient way for the storage and displaying of models, training data, results, in different experiments.
## How to begin?
### Download the project
```
git clone https://github.com/NeverGiveU/CrossNet-pytorch.git
cd CrossNet-pytorch
```
### Dataset
&ensp;&ensp;&ensp;&ensp;Download the dataset of **horse2zebra** in from [Baidu NetDisk](https://pan.baidu.com/s/1BqPsv7E6OwgItjIRxi7CCA).

&ensp;&ensp;&ensp;&ensp;Unzip the .zip file to `datasets`.
### Train
&ensp;&ensp;&ensp;&ensp;Use command `python train.py` to start trainning.
### Test
&ensp;&ensp;&ensp;&ensp;Use command `python test.py` to test after finishing the training.
## Results
&ensp;&ensp;&ensp;&ensp;Some results can be seen in dir `sample`. Notice that, the results from **zebra** to **horse** is better than the results in inverse direction.
![alt sample 1](https://github.com/NeverGiveU/CrossNet-pytorch/tree/master/samples/000010-00.jpg)
![alt sample 2](https://github.com/NeverGiveU/CrossNet-pytorch/tree/master/samples/000061-00.jpg)
![alt sample 3](https://github.com/NeverGiveU/CrossNet-pytorch/tree/master/samples/000089-00.jpg)
![alt sample 4](https://github.com/NeverGiveU/CrossNet-pytorch/tree/master/samples/000116-00.jpg)
![alt sample_1](https://github.com/NeverGiveU/CartoonGAN-pytorch/blob/master/sample_images/0015-001324.jpg)


## Some Notes
&ensp;&ensp;&ensp;&ensp;We found that during the training, the adversarial loss of the generator almostly unchanged, while for descriminator it decreased obviously. We think that is because we trained models in mode of "**G** first and **D** next", in which **D** provided less information for **G** when updating **G**. We will try another training mode of "**D** first and **G** next" in the following days. And updating of project will be done soon.

