# Kilometer-scale single-shot macroscopic Fourier ptychography with meteorology-aware restoration 

This work presents a meteorological-parameter-guided network termed MPGNet (model.py) for camera-array-based single-shot far-field FP imaging and introduces meteorological-parameter-guided regulations into the degradation module (model_TPG-FP.py) for learning-based digital simulation of far-field FP imaging


## Simulation Data Preparation

Please follow this (https://data.vision.ee.ethz.ch/cvl/DIV2K/) to download the DIV2K training dataset.

Please follow this [link from EDSR](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) to download the Flickr2K dataset.


## Download Kilometer-scale Datasets

Please follow this (https://pan.baidu.com/s/1JpufwomI2AMs0cQVaD9zZA?pwd=qqgx 提取码: qqgx) to download the kilometer-scale datasets.


## Download Pre-trained Models

Please follow this (https://pan.baidu.com/s/1JpufwomI2AMs0cQVaD9zZA?pwd=qqgx 提取码: qqgx) to download the pre-trained models.


## Running

**simulation for training data**

```python model_TPG-FP.py --ori_path {path_to_input_folder_for_simulation}```

We provided default parameters in the document (model_TPG-FP.py) for image generation and other applications can further optimize performance by modifying the simulation parameters for TPG-FP relevant to the coherent imaging system.

Please follow this (https://pan.baidu.com/s/1JpufwomI2AMs0cQVaD9zZA?pwd=qqgx 提取码: qqgx) to download the samples of corr_mat and tilt_mat.

**Inference on test data**

```python test_image.py --model_name {path_to_model_for_image_restoration} --test_root {path_to_input_folder_for_sub-aperture_images} --save_root {path_to_save_the_results}```

We provided default parameters in the document (test_image.py) for testing the restoration results of MPGNet on the test data. 


## Acknowledgement

https://github.com/XPixelGroup/BasicSR

https://github.com/microsoft/SeerAttention

https://github.com/Fediory/HVI-CIDNet

https://github.com/xwmaxwma/TinyViM

