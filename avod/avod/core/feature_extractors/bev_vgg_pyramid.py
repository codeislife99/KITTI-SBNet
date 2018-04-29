import tensorflow as tf
import numpy as np
import copy

from avod.core.feature_extractors import bev_feature_extractor

slim = tf.contrib.slim

sbnet_module = tf.load_op_library('/home/allanwan/Classes/16824/proj/KITTI-SBNet/sbnet/sbnet_tensorflow/sbnet_ops/libsbnet.so')


class BevVggPyr(bev_feature_extractor.BevFeatureExtractor):
    """Contains modified VGG model definition to extract features from
    Bird's eye view input using pyramid features.
    """

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def build(self,
              inputs,
              mask_dict,
              input_pixel_size,
              is_training,
              scope='bev_vgg_pyr'):
        """ Modified VGG for BEV feature extraction with pyramid features

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        vgg_config = self.config

        mask = mask_dict['mask']
        base_bcounts = mask_dict['bcounts']
        base_bsize = mask_dict['bsize']
        base_boffset = mask_dict['boffset']
        base_bstride = mask_dict['bstride']
        tol_threshold = 0.25

        num_channels = np.shape(mask)[-1]
        masks_list = [mask]
        bcounts_list = [base_bcounts]
        bsize_list = [base_bsize]
        boffset_list = [base_boffset]
        bstride_list = [base_bstride]
        for iternum in range(3):
            bcounts = bcounts_list[-1]
            bcounts_list.append(bcounts)
            bsize = copy.deepcopy(bsize_list[-1])
            bsize[0] = bsize[0] // 2
            bsize[1] = bsize[1] // 2
            bsize_list.append(bsize)
            boffset = copy.deepcopy(boffset_list[-1])
            boffset[0] = boffset[0] // 2
            boffset[1] = boffset[1] // 2
            boffset_list.append(boffset)
            bstride = copy.deepcopy(bstride_list[-1])
            bstride[0] = bstride[0] // 2
            bstride[1] = bstride[1] // 2
            bstride_list.append(bstride)
            
            mask = tf.identity(mask)
            mask = mask[:, ::2, ::2, :]
            masks_list.append(mask)

        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, 'bev_vgg_pyr', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    # Pad 700 to 704 to allow even divisions for max pooling
                    padded = tf.pad(inputs, [[0, 0], [4, 0], [0, 0], [0, 0]])
                    input_pixel_size[0] = input_pixel_size[0] + 4

                    # Encoder
                    mask = masks_list[0]
                    _bsize = bsize_list[0]
                    _boffset = boffset_list[0]
                    _bstride = bstride_list[0]
                    indices = sbnet_module.reduce_mask(
			mask, tf.constant(bcounts, dtype=tf.int32),
			bsize=_bsize,
			boffset=_boffset,
			bstride=_bstride,
			tol=tol_threshold, # pooling threshold to consider a block as active
			avgpool=True) # max pooling by default

                    """
                    gpu_ops = tf.GPUOptions(allow_growth=True)
                    config = tf.ConfigProto(gpu_options=gpu_ops)
                    sess = tf.Session(config=config)
                    sess = tf.Session()
                    print(sess.run(indices.bin_counts))
                    print(sess.run(indices.active_block_indices))
                    """
                    
                    block_stack = sbnet_module.sparse_gather(
					padded,
					indices.bin_counts,
					indices.active_block_indices,
					bsize=_bsize, # block size
					boffset=_boffset, # block offset
					bstride=_bstride, # block stride
					transpose=False)
                    block_stack = tf.reshape(block_stack, [-1, _bsize[0], _bsize[1], num_channels])
                    operation_output = slim.repeat(block_stack,
                                        vgg_config.vgg_conv1[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv1[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv1')
                    num_channels = vgg_config.vgg_conv1[1]
                    padded = tf.tile(tf.expand_dims(padded[:, :, :, 0], -1), [1, 1, 1, num_channels])
                    conv1 = sbnet_module.sparse_scatter(
					operation_output,
					indices.bin_counts,
					indices.active_block_indices,
					padded, # base tensor to copy to output and overwrite on top of
					bsize=_bsize,
					boffset=_boffset,
					bstride=_bstride,
					add=False,
					atomic=False, # use atomic or regular adds
					transpose=False)
                    conv1 = tf.reshape(conv1, [1, 
					       input_pixel_size[0], 
					       input_pixel_size[1], 
					       num_channels], name='reshape_conv1')
                    pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')

                    #===============================================================

                    mask = masks_list[1]
                    mask = tf.tile(tf.expand_dims(mask[:, :, :, 0], 3), [1, 1, 1, num_channels])
                    input_pixel_size[0] = input_pixel_size[0] // 2
                    input_pixel_size[1] = input_pixel_size[1] // 2
                    _bsize = bsize_list[1]
                    _boffset = boffset_list[1]
                    _bstride = bstride_list[1]
                    indices = sbnet_module.reduce_mask(
			mask, tf.constant(bcounts, dtype=tf.int32),
			bsize=_bsize,
			boffset=_boffset,
			bstride=_bstride,
			tol=tol_threshold, # pooling threshold to consider a block as active
			avgpool=True) # max pooling by default
                    block_stack = sbnet_module.sparse_gather(
					pool1,
					indices.bin_counts,
					indices.active_block_indices,
					bsize=_bsize, # block size
					boffset=_boffset, # block offset
					bstride=_bstride, # block stride
					transpose=False)
                    block_stack = tf.reshape(block_stack, [-1, _bsize[0], _bsize[1], num_channels])
                    operation_output = slim.repeat(block_stack,
                                        vgg_config.vgg_conv2[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv2[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv2')
                    num_channels = vgg_config.vgg_conv2[1]
                    pool1 = tf.tile(tf.expand_dims(pool1[:, :, :, 0], -1), [1, 1, 1, num_channels])
                    conv2 = sbnet_module.sparse_scatter(
					operation_output,
					indices.bin_counts,
					indices.active_block_indices,
					pool1, # base tensor to copy to output and overwrite on top of
					bsize=_bsize,
					boffset=_boffset,
					bstride=_bstride,
					add=False,
					atomic=False, # use atomic or regular adds
					transpose=False)
                    conv2 = tf.reshape(conv2, [1, 
					       input_pixel_size[0], 
					       input_pixel_size[1], 
					       num_channels], name='reshape_conv2')
                    pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

                    #===============================================================

                    mask = masks_list[2]
                    mask = tf.tile(tf.expand_dims(mask[:, :, :, 0], 3), [1, 1, 1, num_channels])
                    input_pixel_size[0] = input_pixel_size[0] // 2
                    input_pixel_size[1] = input_pixel_size[1] // 2
                    _bsize = bsize_list[2]
                    _boffset = boffset_list[2]
                    _bstride = bstride_list[2]
                    indices = sbnet_module.reduce_mask(
			mask, tf.constant(bcounts, dtype=tf.int32),
			bsize=_bsize,
			boffset=_boffset,
			bstride=_bstride,
			tol=tol_threshold, # pooling threshold to consider a block as active
			avgpool=True) # max pooling by default
                    block_stack = sbnet_module.sparse_gather(
					pool2,
					indices.bin_counts,
					indices.active_block_indices,
					bsize=_bsize, # block size
					boffset=_boffset, # block offset
					bstride=_bstride, # block stride
					transpose=False)
                    block_stack = tf.reshape(block_stack, [-1, _bsize[0], _bsize[1], num_channels])
                    operation_output = slim.repeat(block_stack,
                                        vgg_config.vgg_conv3[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv3[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv3')
                    num_channels = vgg_config.vgg_conv3[1]
                    pool2 = tf.tile(tf.expand_dims(pool2[:, :, :, 0], -1), [1, 1, 1, num_channels])
                    conv3 = sbnet_module.sparse_scatter(
					operation_output,
					indices.bin_counts,
					indices.active_block_indices,
					pool2, # base tensor to copy to output and overwrite on top of
					bsize=_bsize,
					boffset=_boffset,
					bstride=_bstride,
					add=False,
					atomic=False, # use atomic or regular adds
					transpose=False)
                    conv3 = tf.reshape(conv3, [1, 
					       input_pixel_size[0], 
					       input_pixel_size[1], 
					       num_channels], name='reshape_conv3')
                    pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')

                    #===============================================================

                    mask = masks_list[3]
                    mask = tf.tile(tf.expand_dims(mask[:, :, :, 0], 3), [1, 1, 1, num_channels])
                    input_pixel_size[0] = input_pixel_size[0] // 2
                    input_pixel_size[1] = input_pixel_size[1] // 2
                    _bsize = bsize_list[3]
                    _boffset = boffset_list[3]
                    _bstride = bstride_list[3]
                    indices = sbnet_module.reduce_mask(
			mask, tf.constant(bcounts, dtype=tf.int32),
			bsize=_bsize,
			boffset=_boffset,
			bstride=_bstride,
			tol=tol_threshold, # pooling threshold to consider a block as active
			avgpool=True) # max pooling by default
                    block_stack = sbnet_module.sparse_gather(
					pool3,
					indices.bin_counts,
					indices.active_block_indices,
					bsize=_bsize, # block size
					boffset=_boffset, # block offset
					bstride=_bstride, # block stride
					transpose=False)
                    block_stack = tf.reshape(block_stack, [-1, _bsize[0], _bsize[1], num_channels])
                    operation_output = slim.repeat(block_stack,
                                        vgg_config.vgg_conv4[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv4[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv4')
                    num_channels = vgg_config.vgg_conv4[1]
                    pool3 = tf.tile(tf.expand_dims(pool3[:, :, :, 0], -1), [1, 1, 1, num_channels])
                    conv4 = sbnet_module.sparse_scatter(
					operation_output,
					indices.bin_counts,
					indices.active_block_indices,
					pool3, # base tensor to copy to output and overwrite on top of
					bsize=_bsize,
					boffset=_boffset,
					bstride=_bstride,
					add=False,
					atomic=False, # use atomic or regular adds
					transpose=False)
                    conv4 = tf.reshape(conv4, [1, 
					       input_pixel_size[0], 
					       input_pixel_size[1], 
					       num_channels], name='reshape_conv4')

                    # Decoder (upsample and fuse features)
                    upconv3 = slim.conv2d_transpose(
                        conv4,
                        vgg_config.vgg_conv3[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv3')

                    concat3 = tf.concat(
                        (conv3, upconv3), axis=3, name='concat3')
                    pyramid_fusion3 = slim.conv2d(
                        concat3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3')

                    upconv2 = slim.conv2d_transpose(
                        pyramid_fusion3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv2')

                    concat2 = tf.concat(
                        (conv2, upconv2), axis=3, name='concat2')
                    pyramid_fusion_2 = slim.conv2d(
                        concat2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion2')

                    upconv1 = slim.conv2d_transpose(
                        pyramid_fusion_2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv1')

                    concat1 = tf.concat(
                        (conv1, upconv1), axis=3, name='concat1')
                    pyramid_fusion1 = slim.conv2d(
                        concat1,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion1')

                    # Slice off padded area
                    sliced = pyramid_fusion1[:, 4:]

                feature_maps_out = sliced

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points
