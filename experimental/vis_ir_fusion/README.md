### Early Fusion of TIR and VIS images

Motivation:
For each fourth timestep of TIR images, a corresponding VIS image (Canon 14mm) exists with higher resolution (3908 x 2600). These images can be classified with established methods such as the OSSP algorithm, which is based on edge detection (https://github.com/wrightni/OSSP).

Approaches:
1) Feed both TIR and VIS images into model for additional information (early fusion) --> ```fuse_train.py```. Use VIS masks obtained from OSSP algorithm.
2) Feed only TIR image into model and use VIS masks obtained from OSSP algorithm.

Requirements:
1) Alignment of TIR and VIS images. Challenging because of different distortions, resolutions, and different field of views. Current approach: remove distortions for each modality as far as possible, manually align upscaled TIR to VIS image in GIMP using rotation, shift and scaling, can not yet be automized because transformation for each image is different --> !too time-consuming for large scale!

Idea:
Check how the required transformation matrix changes for images that are close to each other. If similar enough, perform new manual transformation only every xth image.


Questions:

- Which resolution to take?
    - downscale VIS to IR resolution
        + get good model input size
        - information loss in VIS (?), harder to align
    - upscale IR to VIS
        + easier to align modalities
        - how to get to model input size? downscale again? patch extraction?

- How to extract squares? (+ images have still different sizes)
    - center crop --> information loss
    - patch extraction --> need to find a good patch size for trade off information loss / enough spatial information in single images
    - padding

- How to concatenate information: simple concatenation across channel dimension would result in 4-channel input --> is that still suitable for backbones that are trained on ImageNet aka RGB input?

- How to make image alignment faster (!)