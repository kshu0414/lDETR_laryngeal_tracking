# _l_-DETR for laryngeal tracking

In this repo, we provide our implementation of _l_-DETR model, adapted from DETR model, to predict laryngeal landmark coordinates on videofluoroscopic swallow study (VFSS) images for laryngeal tracking purpose. The model architecture can be found in [model.py](model.py).

## Data Augmentation and Training 
The data preprocess and augmentation steps are in [data_loading.py](data_loading.py). The model is trained following the steps in [train.py](train.py).

## Prediction using Pretrained Weights
For demonstration purpose, pretrained model of a minimal implementation of our _l_-DETR model can be visualized and evaluated using [demo_evaluate.py](demo_evaluate.py).

## Software requirements
Our code was tested on Windows and Linux operating systems with following software requirements:

## Reference
We acknowledge the original DETR codes and posts available from [link](https://github.com/facebookresearch/detr).
- `python` = 3.9.15
- `pytorch` = 1.10.1
- `numpy` = 1.22.1
- `pandas` = 1.4.0
- `scikit-image` = 0.19.1

