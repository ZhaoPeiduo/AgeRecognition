### Age Recognition Using ResNet and Vision Transformer

Disclaimer: Here Age recognition refers to matching faces of the same person across different ages, NOT predicting a person's age given his/her face. 

Dataset: [Cross-Age Celebrity Dataset(CACD2000)](https://bcsiriuschen.github.io/CARC/): Conatining celebrities faces across different ages.

The face crops and the identity mapping can be found in this [link](https://www.kaggle.com/datasets/peiduozhao/cacd-face-crops)

Approach:
- Apply a face detctor to the input image to capture the face and preprocess it to the standard size.
- Initialize backbones with ImageNet pretrained weights
- Gernate training triplets in the following manner: anchor, positive(same id, different age), negative(different id, same age)
- Compute: a) Triplet loss and b) positive-negative cosine embedding loss
- Loss = triplet loss + learnable regularization  factor * cosine embedding loss

TODOs:
- [x] Dataset cleaning (face crop generation)
- [x] Dataset preprocessing (anchor-positive-negative sampling)
- [x] Model definitions
- [x] Loss
- [x] Preprocessor
- [x] Model Training

Age recognizers using resnet101 and vit_b_16 backbones are now avaliable [here](https://drive.google.com/drive/folders/1oG9tei4nwXHCYR-gi-leqKN1TJ2nM7cd?usp=sharing).

### Evaluation Statistics
Vit_b_16: Accuracy = 0.81 at threshold of 0.65 on the test dataset (test_vit_16_b.ipynb)

Resnet101: Accuracy = 0.76 at threshold of 0.65 on the test dataset (test_resnet.ipynb)

### Sample Usage(?)

Using connan anime character as illustrative examples below (where the inspiration for the project is from)

![child_adult](https://github.com/ZhaoPeiduo/AgeRecognition/assets/77187494/de7c2aa3-f5ff-4c41-8f36-803798eb488b)

![child_diffid](https://github.com/ZhaoPeiduo/AgeRecognition/assets/77187494/a7758788-b76f-4170-98e7-2a9f36088ffa)


sample.ipynb shows code samples to use the model for face comparison.

