# Towards a Class-Shielding Data Poisoner
The paper is [here](final_report.pdf).

### Dependencies
Code was executed on an M1 Mac with `tensorflow-macos` 2.12.0, but any Tensorflow 2 should suffice. `tensorflow-metal` version 0.8.0 was used to
    accelerate training.

### Dataset
We used the ILSVRC 2012 ("ImageNet") dataset, which can be downloaded via https://www.tensorflow.org/datasets/catalog/imagenet2012
    
   This download should result in the following directory hierarchy (unused paths omitted):

    ImageNet
        - ILSVRC
            - Annotations
                ...
            - Data
                - CLS-LOC
                    - train
                    - test
                    - val
            - ImageSets
                ...
        - ...
     
   Place the ImageNet directory in the working directory. In order to make the training tractable, we downsampled the data; we recommend that the reader does the same. To do so:

 ```$ python3 make_imagenet_subset.py```
  
which will generate the directory imagenet_subset at the working path, containing 10 symlinks to
    subfolders of ImageNet/ILSVRC/Data/CLS-LOC/train. Training can now be performed:

```$ python3 final_project.py```

final_project.py will expect the imagenet_subset directory created by make_imagenet_subset at the working path, and will also expect resnet_class_lookup.json to identify synset codes for resnet class labels.

Sample execution can be seen in [sample_execution.png](sample_execution.png)

To switch between ResNet and VGG target architectures, comment/uncomment lines 118-138 and change the preprocessor module on line 279 (the last parameter).

