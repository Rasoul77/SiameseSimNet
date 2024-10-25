# SiameseSimNet
A Simple Pytorch Siamese Network for Image Similarity Learning

## Data folder structure
The images shall be organized in a `root_dir` such that there exist a sub-folder per class name, 
```plaintext
root_dir
├── Class1
|   ├── image01.png/jpg
|   ├── image02.png/jpg
├── Class2
|   ├── image01.png/jpg
|   ├── image02.png/jpg
└── Class3
    ├── image01.png/jpg
    ├── image02.png/jpg
```

## Training phase
Modify `config.py` accoding to your application and run `train.py` with the following pattern,

```
python3 train.py --data-path <path to root_dir> \ # Mandatory
                 --class_names Class3,Class2,Class1 \ # Optional - defines a custom order of class indices
                 --use-wandb \ # Optional - whether to use WandB for logging
```

The results of training will be saved into a checkpoint directory named `weights`.

## Inference phase
In order to select a set of representative candidates of each class from the training data, we first run `select_ref_images.py` that creates an output directory with `reference images` that will be used together with the pre-trained model for inference. For an example of how to do inference see [this Kaggle demo notebook](https://www.kaggle.com/code/rasoulmojtahedzadeh/siamese-inference-axial-t2) which is part of the RSNA 2024 Lumbar Spine Degenerative Classification competition.
