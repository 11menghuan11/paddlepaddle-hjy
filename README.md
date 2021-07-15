# ETM-Face Face Detector
## Introduction

ETM-Face is a practical single-stage face detector.

## Data

1. Download the annotations provided by [RetinaFace](https://arxiv.org/abs/1905.00641)(face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [onedrive](https://1drv.ms/u/s!AswpsDO2toNKrjOqf9r9HHzG63jJ?e=a3ifuG)

2. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) and [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) dataset.

3. Organise the WIDERFACE dataset directory under ``ETM-Face/data`` as follows:

```Shell
  data/widerface/    
    train/
      images/
      label.txt
    val/
      images/
      label.txt
    test/
      images/
      label.txt
```
4. Organise the  dataset directory under ``ETM-Face/fddb`` as follows:

```Shell
  fddb/
    FDDB-folds/
    originalPics/
    result1   
```

## Install

1. Install Pytorch with GPU support.
2. Install Deformable Convolution V2 operator from [cvpods](https://https://github.com/Megvii-BaseDetection/cvpods) if you use the DCN based backbone.
3. Requirements:
 ```Shell
   Torch == 1.8.0
   Torchvision == 0.9.0
   Python == 3.8
   NVIDIA GPU == GTX 3090
   Linux CUDA ==11.1
 ```
 ## Training

Please check ``train1.py`` for training.

## Evaluation

1. Download our ETM-Face model [baidu cloud](https://pan.baidu.com/s/1KHZhEXlOnCiXVw9nOVCWgA) (提取码：roy0) trained on WIDER FACE training set to `$DSFD_ROOT/weights/`.

2. Evaluate the trained model via `./widerface_val.py` on WIDER FACE.
```
python widerface_val.py [--trained_model [TRAINED_MODEL]] [--save_folder [SAVE_FOLDER]] 
                         [--widerface_root [WIDERFACE_ROOT]]
    --trained_model      Path to the saved model
    --save_folder        Path of output widerface resutls
    --widerface_root     Path of widerface dataset
```

3. Download the [eval_tool](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip) to show the WIDERFACE performance.

   WiderFace validation mAP: Easy 97.2, Medium 96.5, Hard 91.6.

4. Evaluate the trained model via `./fddb_test.py` on FDDB.
```
python widerface_test.py [--trained_model [TRAINED_MODEL]] [--split_dir [SPLIT_DIR]] 
                         [--data_dir [DATA_DIR]] [--det_dir [DET_DIR]]
    --trained_model      Path of the saved model
    --split_dir          Path of fddb folds
    --data_dir           Path of fddb all images
    --det_dir            Path to save fddb results
```

5. Download the [evaluation](http://vis-www.cs.umass.edu/fddb/evaluation.tgz) to show the FDDB performance.

#### References

https://github.com/Tencent/FaceDetection-DSFD

https://github.com/biubug6/Pytorch_Retinaface

https://github.com/deepinsight/insightface

https://github.com/ultralytics/yolov5

https://github.com/sfzhang15/ATSS





