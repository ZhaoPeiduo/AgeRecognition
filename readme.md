### Age Recognition Using ResNet and Vision Transformer

Dataset: [Cross-Age Celebrity Dataset(CACD2000)](https://bcsiriuschen.github.io/CARC/): Conatining celebrities faces across different ages

The face crops and the identity mapping can be found in this [link](https://drive.google.com/drive/folders/1bv5mg0DhtP4mECQ8Hy7m3pZ5lUPr0MOz?usp=sharing)

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
- [ ] Investigate if vit has overfitted

Age recognizers using resnet101 and vit_b_16 backbones are now avaliable [here](https://drive.google.com/drive/folders/1oG9tei4nwXHCYR-gi-leqKN1TJ2nM7cd?usp=sharing)

Refer to sample.ipynb for sample usage.
