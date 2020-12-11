# PASCAL VOC Instance Segmentation

### Introduction

This is a repo that use mmdetection for instance segmentation task on PASCAL VOC dataset.

### Setup Environment

Clone this repo with submodules

```
git clone --recurse-submodules https://github.com/jinczing/pascal_voc_instance_segmentation.git
cd pascal_voc_instance_segmentation
```



Install requirements

```
pip install mmcv-full==latest+torch1.7.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
%cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
%cd ..
```



Prepare datasets and annotations

Train set: https://drive.google.com/file/d/1HmIArsGNUs2tvsmLBSeXPZoA7OWMRncq/view?usp=sharing

Train annotation: https://drive.google.com/file/d/1m6KWZjQRGJxUpM-5F6VT-GgmBSMPfVDT/view?usp=sharing

Test set: https://drive.google.com/file/d/1k4Wd3lYY0_uNgGjnef96RDuCcoa1XgSu/view?usp=sharing

Test annotation: https://drive.google.com/file/d/1LfOqE5sntdX6yJw8250b9KNNpDcO2F64/view?usp=sharing



Put the unzipped datasets and annotations under mmdetection folder.



Train

```
cd mmdetection
python tools/train.py /path/to/config/file
```

There are three different configurations:

detectors_pointrend_baseline.py (baseline)

detectors_pointrend_resnext.py (change resnet50 to resnext50)

detectors_pointrend_aug.py (training time data augmentation)

You can fine detailed architecture and hyperparameter setups in this(https://drive.google.com/file/d/1RT7SO0qlIPlty_XSIn0tDbXe9VNTiQZF/view?usp=sharing) report

You can also create your own customized configuration file. (please resort to documentation of mmdetection)



### Inference

Pretrained model: https://drive.google.com/file/d/1p_lB2YoB7CiK0Ej8PfoOd6daWC8LU2li/view?usp=sharing

```
python tools/test.py \
path/to/config/file \
path/to/pretrained/model \
--out test.pkl
```



Convert result to submission format

```
pyton convert_result.py \
--in-path path/to/input/pkl/file \
--out-path path/to/output/json/file
```



### Google Colab

Google Colab notebook: https://colab.research.google.com/drive/1uEsGjfRru7x1X8saOMOdsC1CIL08pjKa?usp=sharing

This will help you train and inference without setup your own environment. (please follow the instruction in the notebook).