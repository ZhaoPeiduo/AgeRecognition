### Age Recognition Using ResNet and Vision Transformer

Dataset: [Cross-Age Celebrity Dataset(CACD2000)](https://bcsiriuschen.github.io/CARC/): Conatining celebrities faces across different ages

Approach:
- Apply a face detctor to the input image to capture the face and preprocess it to the standard size.
- Initialize backbones with ImageNet pretrained weights
- Gernate training triplets in the following manner: anchor, positive(same id, different age), negative(different id, same age)
- Compute: a) Triplet loss and b) positive-negative cosine embedding loss
- Loss = weighted(triplet loss + cosine embedding loss)

TODOs:
- [ ] Dataset cleaning (face crop generation)
- [x] Model definitions
- [x] Loss
- [ ] Preprocessor
- [ ] Model Training