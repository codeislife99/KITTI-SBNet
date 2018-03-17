"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings

import keras
import keras_resnet
import keras_resnet.models
import keras_resnet.blocks
import keras_resnet.layers

from ..models import retinanet
import tensorflow as tf
import numpy as np

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)

custom_objects = retinanet.custom_objects.copy()
custom_objects.update(keras_resnet.custom_objects)

allowed_backbones = ['resnet50', 'resnet101', 'resnet152']

parameters = {
    "kernel_initializer": "he_normal"
}

def generate_top_left_mask(xsize, sparsity):
    """
    Generates a square top-left mask with a target sparsity value.

    :param xsize:       [list]      List of 4 int.
    :param sparsity:    [float]     Target sparsity value.

    :return:            [Tensor]    A tensor with shape to be `xsize` and contains a square of 1's
                                    and the rest being 0's.
    """
    density = 1.0 - sparsity
    edge_ratio = np.sqrt(density)
    height = tf.cast(tf.ceil(edge_ratio * xsize[1]), tf.int32)
    width = tf.cast(tf.ceil(edge_ratio * xsize[2]), tf.int32)
    x = tf.Variable(tf.zeros(tf.cast(xsize[:-1], tf.int32)))
    x[:, :height, :width] = 1.0
    return x

def bottleneck_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)
        y = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
        y = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = keras.layers.Conv2D(filters * 4, (1, 1), use_bias=False, name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)
        y = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = keras.layers.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
            shortcut = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f

def ResNet(inputs, blocks, block, include_top=True, classes=1000, freeze_bn=True, numerical_names=None, *args, **kwargs):
    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    xsize = tf.cast(tf.shape(inputs), tf.float32)
    sparsity = 0.5
    mask = generate_top_left_mask(xsize, sparsity)

    x = keras.layers.ZeroPadding2D(padding=3, name="padding_conv1")(inputs)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
    x = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]), freeze_bn=freeze_bn)(x)

        features *= 2

        outputs.append(x)

    if include_top:
        assert classes > 0

        x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
        x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

        return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
    else:
        # Else output each stages features
        return keras.models.Model(inputs=inputs, outputs=outputs, *args, **kwargs)

def ResNet50(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    if blocks is None:
        blocks = [3, 4, 6, 3]
    numerical_names = [False, False, False, False]

    return ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)

def download_imagenet(backbone):
    validate_backbone(backbone)

    backbone = int(backbone.replace('resnet', ''))

    filename = resnet_filename.format(backbone)
    resource = resnet_resource.format(backbone)
    if backbone == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif backbone == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif backbone == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'

    return keras.applications.imagenet_utils.get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def validate_backbone(backbone):
    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))


def resnet_retinanet(num_classes, backbone='resnet50', inputs=None, modifier=None, **kwargs):
    validate_backbone(backbone)

    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet101':
        resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet152':
        resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    # create the full model
    model = retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone=resnet, **kwargs)

    return model


def resnet50_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)


def resnet101_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet101', inputs=inputs, **kwargs)


def resnet152_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet152', inputs=inputs, **kwargs)


def ResNet50RetinaNet(inputs, num_classes, **kwargs):
    warnings.warn("ResNet50RetinaNet is replaced by resnet50_retinanet and will be removed in a future release.")
    return resnet50_retinanet(num_classes, inputs, *args, **kwargs)


def ResNet101RetinaNet(inputs, num_classes, **kwargs):
    warnings.warn("ResNet101RetinaNet is replaced by resnet101_retinanet and will be removed in a future release.")
    return resnet101_retinanet(num_classes, inputs, *args, **kwargs)


def ResNet152RetinaNet(inputs, num_classes, **kwargs):
    warnings.warn("ResNet152RetinaNet is replaced by resnet152_retinanet and will be removed in a future release.")
    return resnet152_retinanet(num_classes, inputs, *args, **kwargs)
