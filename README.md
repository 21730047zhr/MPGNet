# MPGNet: Meteorological-parameters-guided restoration network for camera-array-based Fourier ptychography 

This work presents a meteorological-parameters-guided network termed MPGNet (model.py) for camera-array-based single-shot far-field FP imaging and introduces meteorological-parameters-guided regulations into the degradation module (model_TPG-FP.py) for learning-based digital simulation of far-field FP imaging


## Simulation Data Preparation

Please follow this (https://data.vision.ee.ethz.ch/cvl/DIV2K/) to download the DIV2K training dataset.

Please follow this [link from EDSR](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) to download the Flickr2K dataset.

## Download Pre-trained Models

Please follow this (https://pan.baidu.com/s/1HYm7lhJxFZ1PKB59D5fwgQ?pwd=4d9p 提取码: 4d9p) to download the pre-trained models.

## Running

**simulation for training data**

```python model_TPG-FP.py --ori_path {path_to_input_folder_for_simulation}```

we provided default parameters in the document (model_FP-P2S.py) for image generation and other applications can further optimize performance by modifying the simulation parameters for TPG-FP relevant to the coherent imaging system.

Please follow this (https://pan.baidu.com/s/1HYm7lhJxFZ1PKB59D5fwgQ?pwd=4d9p 提取码: 4d9p) to download the samples of corr_mat and tilt_mat.

**Inference on test data**

```python test_image.py --model_name {path_to_model_for_image_restoration} --test_root {path_to_input_folder_for_sub-aperture_images} --save_root {path_to_save_the_results}```

we provided default parameters in the document (test_image.py) for testing the restoration results of MPGNet on the test data. 

## Acknowledgement

https://github.com/XPixelGroup/BasicSR

https://github.com/microsoft/SeerAttention

https://github.com/Fediory/HVI-CIDNet

https://github.com/xwmaxwma/TinyViM

