# SID 
to install  
- python 3.7.13 (conda create --n xxxx python=3.7)
- pytorch 1.11.0 (conda install -c pytorch pytorch)
- torch-geometric 2.0.4 (conda install pyg -c pyg)
- pip uninstall pytorch-spline-conv (conda)
- rdkit 2020.09.1.0 (conda install -c rdkit rdkit)


# Symmetry-induced Disentanglement on Graphs
![SID model](https://github.com/gmerca/Graph_AE/blob/main/diagram.png "model")

paper:  
Symmetry-induced Disentanglement on Graphs  
Giangiacomo Mercatali, Andre Freitas, Vikas Garg  
https://openreview.net/forum?id=4tM0P_4N8D9

## Model
The paper's model — the Lie-group VAE — is implemented in `classification/Graph_LieVAE.py`.

## Training
Train the compression model:
```
python train_compression.py --m=Lie --d=WattsStrogatz
```
Then train the downstream classifier on its frozen latents:
```
python train_classifier.py --m=Lie --d=WattsStrogatz
```

## Citation
```
@inproceedings{
mercatali2022symmetryinduced,
title={Symmetry-induced Disentanglement on Graphs},
author={Giangiacomo Mercatali and Andre Freitas and Vikas Garg},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=4tM0P_4N8D9}
}
```
