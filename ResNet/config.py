# some training parameters
EPOCHS = 10
BATCH_SIZE = 32
# 三分类任务
NUM_CLASSES = 3
image_height = 32
image_width = 32
channels = 3
save_model_dir = "saved_model/model"
dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"

# choose a network
# model = "resnet34"
model = "resnet50"
# model = "resnet101"
# model = "resnet152"