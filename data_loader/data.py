from os.path import join
from .data_utils import DatasetFromFolder

def get_training_set(dataset, upscale_factor=4, patch_size=64, crop=None):
    train_dir = join(dataset, "RGBD_data_train")

    return DatasetFromFolder(train_dir,
                             upscale_factor=upscale_factor,
                             crop=crop,)


def get_validation_set(dataset, upscale_factor=4):
    validation_dir = join(dataset, "RGBD_data_val")

    return DatasetFromFolder(validation_dir,
                             upscale_factor=upscale_factor)


def get_test_set(dataset, upscale_factor=4):
    test_dir = join(dataset, "RGBD_data_test")

    return DatasetFromFolder(test_dir,
                             upscale_factor=upscale_factor, patch_size=patch_size)
