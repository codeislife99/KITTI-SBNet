"""Detection model trainer.

This runs the DetectionModel trainer.
"""

import argparse
import os
import cv2

import tensorflow as tf
import numpy as np

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel
from avod.core import trainer

tf.logging.set_verbosity(tf.logging.ERROR)


def bev_gen(model_config, train_config, dataset_config):

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)
    dataset.train_val_test = 'train'
    dataset.train_on_all_samples = True
    num_data = dataset.num_samples
    #print(len(dataset.load_sample_names('train')))
    """
    for i in range(num_data):
        outputs = dataset.next_batch(1, shuffle=False)
        output_bundle = outputs[0]
        bev = output_bundle['bev_input'] * 255
        index = output_bundle['sample_name']
        bev_123 = bev[:,:,0:3]
        bev_456 = bev[:,:,3:6]
        cv2.imwrite('bev_samples/' + index + '_0.jpg', bev_123)
        cv2.imwrite('bev_samples/' + index + '_1.jpg', bev_456)
        print(index)
    print(outputs[2])
    output_bundle = outputs[2]
    H = output_bundle['stereo_calib_p2']
    label = output_bundle['label_boxes_3d']
    print(H)
    print(label)
    """
    return


def main(_):
    parser = argparse.ArgumentParser()

    # Defaults
    default_pipeline_config_path = avod.root_dir() + \
        '/configs/avod_cars_example.config'
    default_data_split = 'train'
    default_device = '0'

    parser.add_argument('--pipeline_config',
                        type=str,
                        dest='pipeline_config_path',
                        default=default_pipeline_config_path,
                        help='Path to the pipeline config')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default=default_data_split,
                        help='Data split for training')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default=default_device,
                        help='CUDA device id')

    args = parser.parse_args()

    # Parse pipeline config
    model_config, train_config, _, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            args.pipeline_config_path, is_training=True)

    # Overwrite data split
    dataset_config.data_split = args.data_split

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    bev_gen(model_config, train_config, dataset_config)


if __name__ == '__main__':
    tf.app.run()
