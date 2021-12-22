# Segmentation_SCHP_Tensorflow

- Original project is [here](https://github.com/PeikeLi/Self-Correction-Human-Parsing) implemented in Pytorch


## Dataset structure: 
- Dataset can be downloaded [here](https://drive.google.com/drive/folders/182hTfb-vkTT0desI-ByKsAodv7-olpNB?usp=sharing)
```
Segmentation_SCHP_Tensorflow
+-- ATR 
|   +-- train_images
|   |   +-- xxx.jpg
|   |   +-- ...
|   +-- train_segmentations
|   |   +-- xxx.png
|   |   +-- ...
|   +-- val_images
|   |   +-- xxx.jpg
|   |   +-- ...
|   +-- val_segmentations
|   |   +-- xxx.png
|   |   +-- ...
|   +-- train_id.txt
|   +-- val_id.txt
+-- datasets
+-- log
+-- modules 
+-- networks
+-- utils
+-- Dataset_splitting.py
+-- train.py

```

