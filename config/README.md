# Variable Description Training Config

Clarification of values that are not straight forward.

## Model Configuration

### 'model.architecture'

- **Description**:
- **Possible Values**:

### 'model.backbone'

- **Description**:
- **Possible Values**:

### 'model.im_size'

- **Description**: Patch size used for training
- **Possible Values**:

### 'model.architecture'

- **Description**:
- **Possible Values**:

### 'model.architecture'

- **Description**:
- **Possible Values**:



#parser.add_argument("--im_size", default=480, type=int, choices=[32, 64, 128, 256, 480], help="Patch size to train on. Choices are constrained because of patch extraction setup.")
#parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs. The weights of the best performing training epoch will be stored.")
#parser.add_argument("--loss", default="focal_dice", type=str, choices=["categoricalCE", "focal_dice", "focal"], help="Loss function. E.g. 'categorical_CE' or 'focal_dice'. For more options see sm.")
#parser.add_argument("--backbone", default="resnet34", type=str, help="U-net backbone to use. For options see sm.")
#parser.add_argument("--optimizer", default="Adam", type=str, choices=["Adam", "SGD", "Adamax"], help="Optimizer to use. For options see sm.")
#parser.add_argument("--batch_size", default=2, type=int, help="Batch size. Adjust with respect to training set size and patch size.")
#parser.add_argument("--augmentation_design", default="on_fly", type=str, choices=["none", "offline", "on_fly"], help="Either None, 'offline' (fixed augmentation before training), or 'on_fly' (while feeding data into the model).")
#parser.add_argument("--augmentation_technique", default=4, type=int, choices=[0, 1, 2, 3, 4, 5], help="0 : flip, 1 : rotate, 2 : crop, 3 : brightness contrast, 4 : sharpen blur, 5 : Gaussian noise.")
#parser.add_argument("--augmentation_factor", default=2, type=int, help="Magnitude by which the dataset will be increased through augmentation. Only takes effect when augmentation_design is set 'offline'.")
#parser.add_argument("--use_class_weights", action='store_true', help="If the loss function should account for class imbalance.")
#parser.add_argument("--use_dropout", action='store_true', help="If to use dropout layers after upsampling operations in the decoder.")
#parser.add_argument("--pretrain", default="imagenet", type=str, choices=["imagenet", "none"], help="Either 'imagenet' to use encoder weights pretrained on ImageNet or None to train from scratch.")
#parser.add_argument("--freeze", action='store_true', help="Only takes effect when pretrain is not None. Whether to freeze encoder during training or allow fine-tuning of encoder weights.")