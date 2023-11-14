### Early Fusion of TIR and VIS images

Motivation:
For each fourth timestep of TIR images, a corresponding VIS image (Canon 14mm) exists with higher resolution (3908 x 2600). These images can be classified with established methods such as the OSSP algorithm, which is based on edge detection (https://github.com/wrightni/OSSP).

Approaches:
1) Feed both TIR and VIS images into model for additional information (early fusion) --> ```fuse_train.py```. Use VIS masks obtained from OSSP algorithm.
2) Feed only TIR image into model and use VIS masks obtained from OSSP algorithm.

Requirements:
1) Alignment of TIR and VIS images. Challenging because of different distortions, resolutions, and different field of views. Current approach: remove distortions for each modality as far as possible, manually align upscaled TIR to VIS image in GIMP using rotation, shift and scaling, can not yet be automized because transformation for each image is different.

Idea:
Check how the required transformation matrix changes for images that are close to each other. If similar enough, perform new manual transformation only every xth image.


### Swin Transformer as alternative model

Backbone choice (https://github.com/microsoft/Swin-Transformer):
- use pretrained on ImageNet-22K because Transformer have generally shown to outperform CNN only when trained on large data
- use pretrained on resolution 224 x 224 because this makes non-overlapping patch partitioning possible. Alternative choices would be 256 x 256 or 384 x 384, which would either result in overlapping patches (which has been shown to be a disadvantage in previous experiments), or much information loss.
- desired partitioning method: crop from center and disregard edges which are more prone to distortions
--> Swin-L initial backbone
- Swin Transformer (https://arxiv.org/abs/2111.09883) may be alternative to test later; may result in more accurate results. But: resolution gap or overlapping patches (256 and 384 are available).