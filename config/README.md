# Variable Description Training Config

Explanation of most important config values.

## Model Configuration

#### 'model.architecture'

- **Description**: Model architecture used. 'att_unet' uses attention blocks in the decoder (after upsampling / dropout and before concatenation).
- **Type**: str
- **Possible Values**: 'base_unet', 'att_unet'

#### 'model.backbone'

- **Description**: Backbone used as encoder.
- **Type**: str
- **Possible Values**: see ```models/segmentation_models/segmentation_models/backbones/backbones_factory.py```. In initial experiments, vgg16 and inceptionv3 resulted in worse performance than resnet34.

#### 'model.im_size'
- **Description**: Patch size used for training. Choices are constrained because of patch extraction setup.
- **Type**: int
- **Possible Values**: 32, 64, 128, 256

#### 'model.pretrain'
- **Description**: Either 'imagenet' to use encoder weights pretrained on ImageNet or 'none' to train from scratch.
- **Type**: str
- **Possible Values**: 'imagenet', 'none'

#### 'model.use_dropout'
- **Description**: Whether to use dropout layers in the decoder (after upsampling and before concatenation).
- **Type**: bool
- **Possible Values**: true, false

#### 'model.freeze'
- **Description**: Only takes effect when pretrain is not None. Whether to freeze encoder during training or allow fine-tuning of encoder weights.
- **Type**: bool
- **Possible Values**: true, false

## Augmentation Configuration

#### 'augmentation.design'
- **Description**: Either 'none', 'offline' (fixed augmentation before training), or 'on_fly' (while feeding data into the model).
- **Type**: str
- **Possible Values**: 'on_fly', 'offline', 'none'

#### 'augmentation.technique'
- **Description**: 0 : flip, 1 : rotate, 2 : crop, 3 : brightness contrast, 4 : sharpen blur, 5 : Gaussian noise. Most augmentation techniques resulted in decreasing performance in initial experiments. To add / combine methods, change ```utils/augmentation.py```.
- **Type**: int
- **Possible Values**: 0, 1, 2, 3, 4, 5

#### 'augmentation.factor'
- **Description**: Magnitude by which the dataset will be increased through augmentation. Only takes effect when augmentation_design is set 'offline'.
- **Type**: int


## Training Configuration

#### 'training.use_class_weights'
- **Description**: If the loss function should account for class imbalance.
- **Type**: bool
- **Possible Values**: true, false

#### 'training.num_epochs'
- **Description**: Number of training epochs. The weights of the best performing training epoch will be stored.
- **Type**: int

#### 'training.loss'
- **Description**: Loss function. 
- **Type**: str
- **Possible Values**: 'categoricalCE', 'focal', 'focal_dice'. For more options see sm (must be added in ```utils/train.py```).

#### 'training.optimizer'
- **Description**: Optimizer used.
- **Type**: str
- **Possible Values**: 'Adam', 'Adamax', 'SGD' implemented. For more options see sm (must be added in ```utils/train.py```).

### 'model.batch_size'
- **Description**: Batch size used.
- **Type**: str
- **Possible Values**: Should be small because of small training data. Can be increased for smaller patch sizes.