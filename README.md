# instance-segmentation-on-a-COCO-2017-subset
This project implements instance segmentation on a COCO-2017 subset with severe class imbalance. Two deep learning architectures are compared:

Mask R-CNN: Instance-based segmentation (precise boundaries)
DeepLabV3+: Semantic segmentation (better minority class recall)

Dataset: 300 training images, 300 validation images, 30 test images
Classes: person (88%), cat, sports ball (1.4%), book
Challenge: 63:1 class imbalance ratio
pip install torch torchvision albumentations pycocotools opencv-python
pip install matplotlib seaborn pandas numpy tqdm
```

### 2. Dataset Structure
```
RMDS_Segmentation_Assignment/
├── train-300/
│   ├── data/          # Training images
│   └── labels.json    # COCO format annotations
├── validation-300/
│   ├── data/          # Validation images
│   └── labels.json
└── test-30/           # Test images (no labels)
