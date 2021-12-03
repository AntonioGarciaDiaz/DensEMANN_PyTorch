# The below code is based on the DenseNet-BC implementation by Geoff Pleiss
# https://github.com/gpleiss/efficient_densenet_pytorch,
# itself based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    """
    Defines a bottleneck function for a DenseNet-BC, and outputs it.

    Args:
        norm (nn.Module) - a module that applies 2D batch normalisation.
        relu (nn.Module) - a module that applies the ReLU function.
        conv (nn.Module) - a 2D convolution module (intermediate features).
    """
    # Define the function using the inputed modules.
    def bn_function(*inputs):
        """
        Bottleneck function for a DenseNet-BC.

        Args:
            *inputs (list of Tensor) - a list of inputs for the layer.
        """
        # Concatenate the inputs.
        concated_features = torch.cat(inputs, 1)
        # Apply the modules on the inputs to calculate the output.
        bottleneck_output = conv(relu(norm(concated_features)))
        # Return the output.
        return bottleneck_output

    # Return the newly defined function.
    return bn_function


def variance_scaling_initializer_(tensor, factor=2.0, mode='fan_none',
                                  uniform=False):
    """
    Custom tensor initialiser based on the variance_scaling_initializer
    from old TensorFlow (tf.contrib), coded in the style of PyTorch's
    kaiming_uniform_ and kaiming_normal_. Fills the tensor with values
    according to either a uniform or normal distribution, where the standard
    deviation is based on the square root of a factor / the fan mode.
    The original TensorFlow variance_scaling_initializer may be found at:
    https://github.com/tensorflow/tensorflow/blob/
    86abbaa083beaca05ee32675ac7bfafb58a4557d/
    tensorflow/contrib/layers/python/layers/initializers.py
    PyTorch initialisers kaiming_uniform_ and kaiming_normal_ can be found at:
    https://pytorch.org/docs/stable/_modules/torch/nn/init.html

    Args:
        tensor (Tensor): an n-dimensional torch.Tensor.
        factor (float): a slope factor for calculating the standard deviation
            (default 2.0).
        mode (str): either 'fan_in', 'fan_out', 'fan_avg', or 'fan_none'
            (default). Choosing 'fan_in' preserves the magnitude of the
            variance of the weights in the forward pass. Choosing 'fan_out'
            preserves the magnitudes in the backwards pass. Choosing 'fan_avg'
            corresponds to an average fan between 'fan_in' and 'fan_out'.
            Choosing 'fan_none' does not preserve the magnitude of weights in
            either direction (fan = 1 * filter dimensions).
        uniform (bool): whether to use a uniform or normal distribution.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    if mode == 'fan_none':
        fan = 1
        if tensor.dim() > 2:
            for s in tensor.shape[2:]:
                fan *= float(s)
    else:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        if mode == 'fan_in':
            fan = fan_in
        elif mode == 'fan_out':
            fan = fan_out
        elif mode == 'fan_avg':
            fan = (fan_in + fan_out) / 2.0
    if uniform:
        # Calculate bounds for uniform distribution
        bound = math.sqrt(3.0 * factor / fan)
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)
    else:
        # Calculate standard deviation for normal distribution
        std = math.sqrt(1.3 * factor / fan)
        with torch.no_grad():
            return tensor.normal_(0, std)


class _DenseLayer(nn.Module):
    """
    Dense layer class to be used inside instances of _DenseBlock.

    Args:
        num_input_features (int) - number of input feature channels.
        growth_rate (int) - number of filters the layer (`k` in paper).
        bc_mode (bool) - whether or not to use the DenseNet-BC architecture.
        bn_size (int) - for DenseNet-BC, multiplicative factor for filters in
            bottleneck layers.
        bn_growth_rate (int) - for DenseNet-BC, number of filters in bottleneck
            layers (to be multiplied by bn_size for the actual number).
        drop_rate (float) - dropout rate after each dense layer.
        efficient (bool) - set to True to use checkpointing (default False).

    Attributes:
        norm1 (nn.Module) - a module that applies 2D batch normalisation.
        relu1 (nn.Module) - a module that applies the ReLU function.
        conv1 (nn.Module) - a 2D convolution module, with kernel_size=1 for
            DenseNet-BC (intermediate features) and kernel_size=3 for DenseNet.
        norm2 (nn.Module) - for DenseNet-BC, a second 2D batch normalisation.
        relu2 (nn.Module) - for DenseNet-BC, a second ReLU function.
        conv2 (nn.Module) - for DenseNet-BC, a 2D convolution with
            kernel_size=3 (final features).
        num_input_features (int) - from args.
        num_filters (int) - number of filters in the layer's last convolution
            (corresponds to the number of 2D outputs, i.e. initialy
            growth_rate from args).
        num_bn_filters (int) - for DenseNet-BC, number of filters in the
            bottleneck layer's convolution (i.e. bn_size * bn_growth_rate).
        bc_mode (bool) - from args.
        drop_rate (float) - from args.
        efficient (bool) - from args.
    """
    def __init__(self, num_input_features, growth_rate, bn_size,
                 bn_growth_rate, bc_mode, drop_rate, efficient=False):
        """
        Initialiser for the _DenseLayer class.
        """
        super(_DenseLayer, self).__init__()

        # Initial operations (normalisation and ReLU).
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        if bc_mode:
            # DenseNet-BC architecture (contains an extra convolution).
            self.add_module(
                'conv1', nn.Conv2d(num_input_features,
                                   bn_size * bn_growth_rate, kernel_size=1,
                                   stride=1, bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * bn_growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module(
                'conv2', nn.Conv2d(bn_size * bn_growth_rate, growth_rate,
                                   kernel_size=3, stride=1, padding=1,
                                   bias=False)),
        else:
            # DenseNet architectures.
            self.add_module(
                'conv1', nn.Conv2d(num_input_features, growth_rate,
                                   kernel_size=3, stride=1, padding=1,
                                   bias=False)),

        # Copy interesting args as class attributes.
        self.num_input_features = num_input_features
        self.num_filters = growth_rate
        if bc_mode:
            self.num_bn_filters = bn_size * bn_growth_rate
        self.bc_mode = bc_mode
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        """
        The forward pass for the dense layer.

        Args:
            *prev_features (list of Tensor) - the input features for the layer
                (outputs of all previous layers + global input).

        Returns:
            new_features (Tensor) - the layer's output features.
        """
        # The forward pass is different for DenseNet and for DenseNet-BC.
        if self.bc_mode:
            # Use _bn_function_factory to create the bottleneck function.
            bn_function = _bn_function_factory(self.norm1, self.relu1,
                                               self.conv1)
            # If requested and possible, use checkpoints for memory efficience.
            if self.efficient and any(prev_feature.requires_grad
                                      for prev_feature in prev_features):
                bottleneck_output = cp.checkpoint(bn_function, *prev_features)
            else:
                bottleneck_output = bn_function(*prev_features)
            # Calculate the new features from the bottleneck output.
            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output
                                                            )))
        else:
            # Calculate the new features from the concat. previous features.
            new_features = self.conv1(self.relu1(self.norm1(
                torch.cat(prev_features, 1))))
        # Apply dropout if requested.
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return new_features

    def get_kCS_list(self):
        """
        Return the connection strenght values for each filter in the layer
        (i.e. each of the filters from the layer's last Conv2d).
        """
        kCS_list = []
        # Get the right Conv2d (in bc_mode it is the one after the bottleneck).
        if self.bc_mode:
            filter_weight_data = self.conv2.weight.data
        else:
            filter_weight_data = self.conv1.weight.data
        weights_per_filter = np.prod(filter_weight_data.size()[1:])
        filter_weight_data = filter_weight_data.tolist()
        # The first dimension is that of filters (= output channels).
        for f in range(len(filter_weight_data)):
            # kCS = sum of all weights in a filter / number of weights.
            kCS_value = 0
            for kernel in filter_weight_data[f]:
                for row in kernel:
                    for pixel in row:
                        kCS_value += abs(pixel)
            kCS_value = kCS_value/weights_per_filter
            kCS_list.append(kCS_value)
        # Return the complete list of kCS values
        return kCS_list

    def complementarity(self, old_weights, num_new_filters=1, patience=None):
        """
        Complementarity mechanism for filter additions in the layer.
        Given the weights of existing filters, and the number of new filters
        to add, generates a sign distribution where the new filters have got
        opposite (flipped) signs to the existing filters with lowest kCS.
        This ensures that new filters are complementary to existing ones, and
        "help" or "compensate" in some way for low-kCS filters.

        Args:
            old_weights (Tensor) - the weights for existing filters.
            num_new_filters (int) - number of filters to add (default 1).
            patience (int or None) - maximum number of alternativee sign
                distributions to try if the new filter's sign distribution
                already exists in the layer. If None, it is the number of
                features in a filter (which is also the number of signs in it).

        Returns:
            filter_signs (Tensor) - the new sign distribution for the layer.
        """
        # The filters' number of input features depends on bc_mode.
        input_features = (self.num_bn_filters if self.bc_mode
                          else self.num_input_features)
        # Default patience is the number of features (and signs) in a filter.
        if patience is None:
            patience = 9*input_features  # 3*3*input_features

        # Get the sign distribution of all the filters already in the layer.
        filter_signs = []
        for old_filter in old_weights:
            filter_signs.append(old_filter.sign())
        # Get the ids of the filters with the lowest CS.
        kCS_list = self.get_kCS_list()
        compl_filters = sorted(
            range(len(kCS_list)), key=lambda i: kCS_list[i]
            )[:num_new_filters]

        # Generate the sign distribution for each new filter to be added:
        # they must be the opposite of those of the filters with lowest CS.
        for new_f in range(num_new_filters):
            new_f_signs = -1*filter_signs[compl_filters[new_f]]
            # Check if sign distribution already exists
            new_f_signs_try = new_f_signs
            sign_distr_exists = True
            while sign_distr_exists and patience > 0:
                # Compare with each of the distributions
                sign_distr_exists = False
                for sign_distr in filter_signs:
                    sign_distr_exists = sign_distr_exists or (
                        new_f_signs_try == sign_distr).all()
                # If so, try with one of the signs switched randomly
                if sign_distr_exists:
                    new_f_signs_try = torch.clone(new_f_signs)
                    random_sign = tuple(torch.LongTensor([
                        np.random.randint(input_features),
                        np.random.randint(3), np.random.randint(3)]))
                    new_f_signs_try.index_put_(
                        random_sign, -1*new_f_signs_try[
                            random_sign[0]][random_sign[1]][random_sign[2]])
                    patience -= 1
            # Add the new sign distribution to the list
            filter_signs.append(new_f_signs_try)

        # Return the list of sign distributions as a single tensor.
        return torch.stack(filter_signs)

    def add_new_filters(self, num_new_filters=1, complementarity=True):
        """
        Adds new filters to the layer.
        Actually replaces the layer's last convolution with a new one that has
        got one more filter, and copies the weight values from the previous
        last convolution into the new one.

        Args:
            num_new_filters (int) - number of filters to add (default 1).
            complementarity (bool) - whether or not new filters should be
                initialised using the complementarity mechanism (default True).
        """
        # The filters' number of input features depends on bc_mode.
        input_features = (self.num_bn_filters if self.bc_mode
                          else self.num_input_features)
        # Create the new convolution, and initialise its weights.
        new_conv = nn.Conv2d(input_features,
                             self.num_filters + num_new_filters,
                             kernel_size=3, stride=1, padding=1, bias=False)
        variance_scaling_initializer_(new_conv.weight.data)

        # Copy the weights from the old convolution to the new one.
        old_conv_weight = (self.conv2.weight.data.cpu() if self.bc_mode
                           else self.conv1.weight.data.cpu())
        old_conv_indexes = [i for i in range(self.num_filters)]
        new_conv.weight.data.index_copy_(
            0, torch.as_tensor(old_conv_indexes), old_conv_weight)

        # If using the complementarity mechanism, make each new filter's sign
        # distribution complementary to that of a filter with low kCS.
        if complementarity:
            filter_signs = self.complementarity(old_conv_weight,
                                                num_new_filters)
            new_conv.weight.data.copysign_(filter_signs)

        # Replace the old convolution with the new one.
        if self.bc_mode:
            self.conv2 = new_conv
        else:
            self.conv1 = new_conv
        # Update the number of filters.
        self.num_filters += num_new_filters

    def remove_filters(self, filter_ids):
        """
        Removes specific filters in the layer.
        Actually replaces the layer's last convolution with a new one where the
        number of filters is self.num_filters - len(filter_ids), and copies the
        weight values from the filters that should be kept in the previous last
        convolution into the new convolution.

        Args:
            filter_ids (list of int) - identifiers for the filters to remove.
        """
        # The filters' number of input features depends on bc_mode.
        input_features = (self.num_bn_filters if self.bc_mode
                          else self.num_input_features)
        # Create the new convolution.
        new_conv = nn.Conv2d(input_features,
                             self.num_filters - len(filter_ids),
                             kernel_size=3, stride=1, padding=1, bias=False)

        # Copy the weights from the old convolution to the new one.
        old_conv_weight = (self.conv2.weight.data.cpu() if self.bc_mode
                           else self.conv1.weight.data.cpu())
        old_conv_indexes = [
            i for i in range(self.num_filters) if i not in filter_ids]
        new_conv.weight.data.copy_(old_conv_weight[old_conv_indexes])

        # Replace the old convolution with the new one.
        if self.bc_mode:
            self.conv2 = new_conv
        else:
            self.conv1 = new_conv
        # Update the number of filters.
        self.num_filters -= len(filter_ids)


class _Transition(nn.Sequential):
    """
    Transition class to be used inside instances of DenseNet, between two
    instances of _DenseBlock.

    As it inherits from nn.Sequential (rather than nn.Module), the forward pass
    does not need to be defined. All of its nn.Module attributes are executed
    in the order in which they are created.

    Args:
        num_input_features (int) - number of input feature channels.
        num_output_features (int) - number of output feature channels.

    Attributes:
        norm (nn.Module) - a module that applies 2D batch normalisation.
        relu (nn.Module) - a module that applies the ReLU function.
        conv (nn.Module) - a 2D convolution module with kernel size=1 (here is
            where compression/reduction is applied for DenseNet-BC).
        pool (nn.Module) - a module that applies 2D average pooling.
    """
    def __init__(self, num_input_features, num_output_features):
        """
        Initialiser for the _Transition class.
        """
        super(_Transition, self).__init__()
        # Operations: batch norm., ReLU, conv. with kernel size 1, avg. pool.
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
                                          num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    """
    Dense block class to be used inside instances of DenseNet.

    Args:
        num_layers (int) - number of layers inside the block.
        num_input_features (int) - number of input feature channels.
        growth_rate (int) - number of filters in each layer (`k` in paper).
        layer_config (list of int or None) - optional list containing a
            specific number of filters for each layer.
        update_growth_rate (bool) - whether or not to change the growth_rate
            before adding each layer, using the previous layer's final number
            of filters as the new value.
        bc_mode (bool) - whether or not to use the DenseNet-BC architecture.
        bn_size (int) - multiplicative factor for bottleneck layers.
        drop_rate (float) - dropout rate after each dense layer.
        efficient (bool) - set to True to use checkpointing (default False).

    Attributes:
        num_layers (int) - from args.
        num_input_features (int) - from args.
        num_output_features (int) - number of output feature channels.
        layer_config (int) - from args (or deduced from them if None in args).
        efficient (bool) - from args, becomes default for layer additions.
        *denselayerN (nn.Module) - Nth dense layer in the block.
    """
    def __init__(self, num_layers, num_input_features, growth_rate,
                 layer_config, update_growth_rate, bc_mode, bn_size, drop_rate,
                 efficient=False):
        """
        Initialiser for the _DenseBlock class.
        """
        super(_DenseBlock, self).__init__()
        # Keep a reference to the number of layers.
        self.num_layers = num_layers
        # Keep the numbers of input and output features as a reference.
        self.num_input_features = num_input_features
        self.num_output_features = num_input_features + (
            num_layers*growth_rate if layer_config is None
            else sum(layer_config))
        # Keep the layer configuration.
        self.layer_config = ([growth_rate for i in range(num_layers)]
                             if layer_config is None else layer_config)
        # Keep the efficient argument (to set defaults).
        self.efficient = efficient

        # Create each layer in the dense block.
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + sum(self.layer_config[:i]),
                growth_rate=self.layer_config[i],
                bc_mode=bc_mode,
                bn_size=bn_size,
                bn_growth_rate=(growth_rate if not update_growth_rate
                                or i == 0 else self.layer_config[i-1]),
                drop_rate=drop_rate,
                efficient=efficient
            )
            # Add each layer to the _DenseBlock
            # (as a Module, it can contain other Module objects as children).
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        """
        The forward pass for the dense block.

        Args:
            init_features (Tensor) - the initial (global) input for the block.

        Returns:
            (Tensor) - a concatenation of (1) the outputs of all layers in the
                block and (2) the block's global input.
        """
        # Initialise a list that will contain the outputs of all layers.
        features = [init_features]
        # For each layer, calculate the output and add it to the list.
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        # The output is a concatenation of all outputs + the global input.
        return torch.cat(features, 1)

    def get_kCS_list_from_layer(self, l):
        """
        Return the connection strenght values for each filter in a given layer.

        Args:
            l (int) - identifier number for the layer inside the block.
        """
        # Account for negative block and layer IDs, etc.
        true_l = l % self.num_layers
        # Get layer l of the block, then get the kCS for each filter.
        requested_layer_name = "self.denselayer%d" % (true_l+1)
        kCS_list = []
        exec("kCS_list.extend(%s.get_kCS_list())" % requested_layer_name)
        return kCS_list

    def add_new_filters(self, num_new_filters=1, complementarity=True):
        """
        Adds new filters to the dense block's last layer.

        Args:
            num_new_filters (int) - number of filters to add (default 1).
            complementarity (bool) - whether or not new filters should be
                initialised using the complementarity mechanism (default True).
        """
        # Reconstruct the block's last layer with the new filters.
        exec(("self.denselayer{}.add_new_filters("
              + "num_new_filters={}, complementarity={})").format(
            self.num_layers, num_new_filters, complementarity))
        # Update the layer_config and the number of output features.
        self.layer_config[-1] += num_new_filters
        self.num_output_features += num_new_filters

    def remove_filters(self, filter_ids):
        """
        Removes specific filters in the dense block's last layer.

        Args:
            filter_ids (list of int) - identifiers for the filters to remove.
        """
        # Reconstruct the block's last layer with the specific filters removed.
        exec(("self.denselayer{}.remove_filters(filter_ids={})").format(
                self.num_layers, filter_ids))
        # Update the layer_config and the number of output features.
        self.layer_config[-1] -= len(filter_ids)
        self.num_output_features -= len(filter_ids)

    def add_new_layers(self, growth_rate, bc_mode, bn_size, bn_growth_rate,
                       drop_rate, num_new_layers=1, efficient=None):
        """
        Adds new layers to the dense block.

        Args:
            growth_rate (int) - number of filters in each layer (`k` in paper).
            bc_mode (bool) - whether or not to use DenseNet-BC.
            bn_size (int) - multiplicative factor for bottleneck layers.
            bn_growth_rate (int) - for DenseNet-BC, number of filters in
                bottleneck layers (to be multiplied by bn_size for the actual
                number).
            drop_rate (float) - dropout rate after each dense layer.
            num_new_layers (int) - number of layers to add (default 1).
            efficient (bool) - set to True to use checkpointing
                (default None, i.e. use the value provided at creation).
        """
        # Handle none arguments.
        if efficient is None:
            efficient = self.efficient

        # Create each layer to be added.
        for i in range(num_new_layers):
            layer = _DenseLayer(
                self.num_output_features + i*growth_rate,
                growth_rate=growth_rate,
                bc_mode=bc_mode,
                bn_size=bn_size,
                bn_growth_rate=bn_growth_rate,
                drop_rate=drop_rate,
                efficient=efficient
            )
            # Initialise the new layer's weights.
            for name, param in layer.named_parameters():
                if 'conv' in name and 'weight' in name:
                    if bc_mode and 'conv1' in name:
                        variance_scaling_initializer_(param.data,
                                                      mode='fan_in')
                    else:
                        variance_scaling_initializer_(param.data,
                                                      mode='fan_none')
                elif 'norm' in name and 'weight' in name:
                    param.data.fill_(1)
                elif 'norm' in name and 'bias' in name:
                    param.data.fill_(0)
            # Add the new layer to the _DenseBlock.
            self.add_module(
                'denselayer%d' % (self.num_layers + i + 1), layer)

        # Update the number of layers, layer_config and output features.
        self.num_layers += num_new_layers
        self.layer_config.extend(
            [growth_rate for new_l in range(num_new_layers)])
        self.num_output_features += num_new_layers*growth_rate


class DenseNet(nn.Module):
    """
    Densenet model class, based on "Densely Connected Convolutional Networks"
    (https://arxiv.org/pdf/1608.06993.pdf).

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper).
        block_config (list of int or list of list of int) - number of layers in
            each pooling block, and optionally number of filters in each layer
            (default [16, 16, 16]).
        update_growth_rate (bool) - whether or not to update the DenseNet's
            growth rate attribute before each layer/filter addition, using the
            previous layer's final number of filters as the new value
            (default True).
        bc_mode (bool) - whether or not to use the DenseNet-BC architecture,
            with a bottleneck after each convolution + compression at
            transition layers (default True).
        reduction (float) - reduction (theta) of the number of parameters
            during compression at transition layers in DenseNet-BC
            (between 0 and 1, default 0.5).
        num_init_features (int) - number of filters to learn in the first
            convolution layer (default 24).
        bn_size (int) - multiplicative factor for number of bottleneck layers
            (i.e. bn_size * k features in the bottleneck layer) (default 4).
        drop_rate (float) - dropout rate after each dense layer (default 0).
        num_classes (int) - number of classification classes (default 10).
        small_inputs (bool) - set to True if images are 32x32, otherwise
            assumes images are larger (default True).
        efficient (bool) - set to True to use checkpointing (much more memory
            efficient, but slower) (default False).
        seed (int or None) - optional seed for the random number generator
            (default None).

    Attributes:
        features (nn.Sequential) - contains the operations that form the
                convolutional part of the DenseNet:
            -> conv0 (and optionally norm0, relu0 and pool0) (nn.Module) -
                the initial convolution and related operations.
            -> *denseblockN (_DenseBlock) - the Nth dense block.
            -> *transitionN (_Transition) - the Nth transition between blocks.
            -> norm_final (nn.Module) - the final 2D batch norm. module.
        classifier (nn.Linear) - the fully-connected layer at the end of the
            DenseNet, for outputting class predictions.
        growth_rate (int) - from args.
        block_config (list of int) - from args, but only containing the number
            of layers in each pooling block.
        update_growth_rate (bool) - from args.
        bc_mode (bool) - from args.
        reduction (float) - from args.
        bn_size (int) - from args.
        drop_rate (float) - from args.
        num_classes (int) - from args.
        efficient (bool) - from args, becomes default for layer additions.
    """
    def __init__(self, growth_rate=12, block_config=[16, 16, 16],
                 update_growth_rate=True, bc_mode=True, reduction=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False,
                 seed=None):
        """
        Initialiser for the DenseNet class.

        Raises:
            AssertionError: reduction should be between 0 and 1.
        """
        super(DenseNet, self).__init__()

        # Copy interesting args as class attributes.
        self.growth_rate = growth_rate
        if type(block_config[0]) == list:
            self.block_config = [len(l) for l in block_config]
        else:
            self.block_config = block_config
        self.update_growth_rate = update_growth_rate
        self.bc_mode = bc_mode
        self.reduction = reduction
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.efficient = efficient
        # Handle reduction value for compression.
        if self.bc_mode:
            assert 0 < reduction <= 1, 'reduction should be between 0 and 1.'
            self.reduction = reduction
        else:
            self.reduction = 1

        # Set seed manually if required.
        if seed is not None:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # Create the first convolution (before dense blocks).
        # The "features" Sequential is created at the same time.
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3,
                                    stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7,
                                    stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                       ceil_mode=False))
            ]))

        # Create each individual dense block.
        self.num_features = num_init_features
        for i, num_layers in enumerate(self.block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=self.num_features,
                growth_rate=self.growth_rate,
                layer_config=(block_config[i] if type(block_config[i]) == list
                              else None),
                update_growth_rate=update_growth_rate,
                bc_mode=bc_mode,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient
            )
            # Add each new dense block to the Sequential.
            self.features.add_module('denseblock%d' % (i + 1), block)
            # Update the number of output features.
            if type(block_config[i]) == list:
                self.num_features += sum(block_config[i])
                # Optionally also update the growth rate
                if self.update_growth_rate:
                    self.growth_rate = block_config[i][-1]
            else:
                self.num_features += num_layers * self.growth_rate
            # Create a transition layer for all dense blocks except the last.
            if i != len(self.block_config) - 1:
                trans = _Transition(num_input_features=self.num_features,
                                    num_output_features=int(
                                        self.num_features * reduction))
                # Add the transition layer to the Sequential.
                self.features.add_module('transition%d' % (i + 1), trans)
                # Update the number of output features.
                self.num_features = int(self.num_features * reduction)

        # Create the final batch norm., and add it to the Sequential.
        self.features.add_module(
            'norm_final', nn.BatchNorm2d(self.num_features))

        # Create the classifier (linear layer).
        self.classifier = nn.Linear(self.num_features, self.num_classes)

        # Initialisation.
        # print("\nParameter name list:\n")
        for name, param in self.named_parameters():
            # print(name)
            if 'conv' in name and 'weight' in name:
                if bc_mode and 'conv1' in name:
                    variance_scaling_initializer_(param.data, mode='fan_in')
                else:
                    variance_scaling_initializer_(param.data, mode='fan_none')
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        """
        The forward pass for the DenseNet.

        Args:
            x (Tensor) - The input data.

        Returns:
            out (Tensor) - The output data.
        """
        # Pass the data to the "features" Sequential (a sequence of modules).
        features = self.features(x)
        # Pass the output through ReLU and adaptive average pooling.
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # Finally, flatten the output and pass it to the linear classifier.
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def get_kCS_list_from_layer(self, b, l):
        """
        Return the connection strenght values for each filter in a given layer.

        Args:
            b (int) - identifier number for the block.
            l (int) - identifier number for the layer inside the block.
        """
        # Account for negative block and layer IDs, etc.
        true_b = b % len(self.block_config)
        true_l = l % self.block_config[true_b]
        # Get layer l of block b, and check if it is split into filters.
        # If so, get the conection strength for each filter in that layer.
        requested_block_name = "self.features.denseblock%d" % (true_b+1)
        kCS_list = []
        exec("kCS_list.extend(%s.get_kCS_list_from_layer(%d))" % (
            requested_block_name, true_l))
        return kCS_list

    def reconstruct_transition_to_classes(self, preserve_transition=True,
                                          filter_ids=None):
        """
        Reconstruct the transition layer to classes (final BatchNorm2D and
        classifier) after adding a new filter or layer in the last block.
        The transition layer may be completely new, or preserved mostly
        unchanged except for the new weights.

        Args:
            preserve_transition (bool) - whether or not to preserve the
                transition to classes (default True).
            filter_ids (list of int or None) - if preserving the transition to
                classes, optional list of filters in that transition that
                should be removed post-pruning.
        """
        # Update the num_features (will be used for the transition to classes).
        old_num_features = self.num_features
        exec(("self.num_features = "
              + "self.features.denseblock%d.num_output_features") % len(
                self.block_config))

        # Create and initialise the new transition to classes.
        new_norm_final = nn.BatchNorm2d(self.num_features)
        new_norm_final.weight.data.fill_(1)
        new_norm_final.bias.data.fill_(0)
        new_classifier = nn.Linear(self.num_features, self.num_classes)
        new_classifier.weight.data.fill_(1)
        new_classifier.bias.data.fill_(0)

        # If required, copy the data from the old transition to classes.
        if preserve_transition:
            # Identify which filter indexes to keep, and where to copy them.
            if filter_ids is None:
                copy_indexes = [i for i in range(old_num_features)]
                keep_indexes = copy_indexes
            else:
                copy_indexes = [
                    i for i in range(old_num_features - len(filter_ids))]
                keep_indexes = [
                    i for i in range(old_num_features) if i not in filter_ids]
            # Batchnorm weights and biases.
            bn_weight = self.features.norm_final.weight.data.cpu()
            new_norm_final.weight.data.index_copy_(
                0, torch.as_tensor(copy_indexes), bn_weight[keep_indexes])
            bn_bias = self.features.norm_final.bias.data.cpu()
            new_norm_final.bias.data.index_copy_(
                0, torch.as_tensor(copy_indexes), bn_bias[keep_indexes])
            # Classifier weights and biases.
            cl_weight = self.classifier.weight.data.cpu()
            new_classifier.weight.data.index_copy_(
                1, torch.as_tensor(copy_indexes), cl_weight[:, keep_indexes])
            new_classifier.bias.data.copy_(self.classifier.bias.data.cpu())

        self.features.norm_final = new_norm_final
        self.classifier = new_classifier

    def add_new_filters(self, num_new_filters=1, complementarity=True,
                        preserve_transition=True):
        """
        Adds new filters to the last layer of the last dense block in the
        DenseNet.

        Args:
            num_new_filters (int) - number of filters to add (default 1).
            complementarity (bool) - whether or not new filters should be
                initialised using the complementarity mechanism (default True).
            preserve_transition (bool) - whether or not to preserve the
                transition to classes (final BatchNorm2D and classifier)
                (default True).
        """
        # Execute the command to add the new filters (in the right block).
        exec(("self.features.denseblock{}.add_new_filters("
              + "num_new_filters={}, complementarity={})").format(
            len(self.block_config), num_new_filters, complementarity))

        # Reconstruct the transition to classes.
        self.reconstruct_transition_to_classes(
            preserve_transition=preserve_transition)

    def remove_filters(self, filter_ids, preserve_transition=True):
        """
        Removes specific filters in the last layer of the last dense block in
        the DenseNet.

        Args:
            filter_ids (list of int) - identifiers for the filters to remove.
            preserve_transition (bool) - whether or not to preserve the
                transition to classes (final BatchNorm2D and classifier)
                (default True).
        """
        # Execute the command to remove the filters (in the right block).
        exec(("self.features.denseblock{}.remove_filters("
              + "filter_ids={})").format(
                len(self.block_config), filter_ids))

        # Reconstruct the transition to classes.
        self.reconstruct_transition_to_classes(
            preserve_transition=preserve_transition, filter_ids=filter_ids)

    def add_new_layers(self, num_new_layers=1, growth_rate=None,
                       preserve_transition=True, efficient=None):
        """
        Adds new layers to the last dense block in the DenseNet.

        Args:
            num_new_layers (int) - number of layers to add (default 1).
            growth_rate (int or None) - number of filters in the new layers,
                (default None, i.e. the DenseNet's growth_rate attribute).
            preserve_transition (bool) - whether or not to preserve the
                transition to classes (final BatchNorm2D and classifier)
                (default True).
            efficient (bool) - set to True to use checkpointing
                (default None, i.e. use the value provided at creation).
        """
        # Before any operations, update the growth rate value if required.
        if self.update_growth_rate:
            exec("self.growth_rate = self.features.denseblock{}."
                 "layer_config[-1]".format(len(self.block_config)))

        # Handle None arguments.
        if growth_rate is None:
            growth_rate = self.growth_rate
        if efficient is None:
            efficient = self.efficient

        # Execute the command to add the new layers (in the right dense block).
        exec("self.features.denseblock{}.add_new_layers(growth_rate, "
             "self.bc_mode, self.bn_size, self.growth_rate, self.drop_rate, "
             "num_new_layers=num_new_layers, efficient=efficient)".format(
                len(self.block_config)))
        # Update the block_config.
        self.block_config[-1] += num_new_layers

        # Reconstruct the transition to classes.
        self.reconstruct_transition_to_classes(
            preserve_transition=preserve_transition)

    def add_new_block(self, num_layers=1, growth_rate=None, efficient=None):
        """
        Add a transition layer, and a new block (with one layer) at the end
        of the current last block. The number of layers and growth rate for
        that block are as specified in the args.

        Args:
            num_layers (int) - number of layers in the new block (default 1).
            growth_rate (int or None) - number of filters in the new layers,
                (default None, i.e. the DenseNet's growth_rate attribute).
            efficient (bool) - set to True to use checkpointing
                (default None, i.e. use the value provided at creation).
        """
        # Before any operations, update the growth rate value if required.
        if self.update_growth_rate:
            exec("self.growth_rate = self.features.denseblock{}."
                 "layer_config[-1]".format(len(self.block_config)))

        # Handle None arguments.
        if growth_rate is None:
            growth_rate = self.growth_rate
        if efficient is None:
            efficient = self.efficient

        # Remove the final batch norm. from the Sequential.
        del self.features.norm_final

        # Create a transition layer for the current last dense block.
        trans = _Transition(num_input_features=self.num_features,
                            num_output_features=int(
                                self.num_features * self.reduction))
        # Add the transition layer to the Sequential.
        self.features.add_module('transition%d' % (len(self.block_config)),
                                 trans)
        # Update the number of output features.
        self.num_features = int(self.num_features * self.reduction)

        # Create the new block and add it to the Sequential.
        block = _DenseBlock(
            num_layers=num_layers,
            num_input_features=self.num_features,
            growth_rate=growth_rate,
            bc_mode=self.bc_mode,
            bn_size=self.bn_size,
            drop_rate=self.drop_rate,
            efficient=efficient
        )
        self.features.add_module('denseblock%d' % (len(self.block_config)+1),
                                 block)
        # Update the number of output features.
        self.num_features = self.num_features + num_layers * growth_rate

        # Update the block_config.
        self.block_config.append(num_layers)

        # Create and add a new transition to classes.
        # Recreate the final batch norm. and classifier.
        self.features.add_module('norm_final',
                                 nn.BatchNorm2d(self.num_features))
        self.classifier = nn.Linear(self.num_features, self.num_classes)
        # Initialise their values.
        # N.B.: No weight or bias data is preseved because adding a new block
        # means that the output features will be completely different.
        self.features.norm_final.weight.data.fill_(1)
        self.features.norm_final.bias.data.fill_(0)
        self.classifier.weight.data.fill_(1)
        self.classifier.bias.data.fill_(0)

    def count_trainable_params(self):
        """
        Count the total number of trainable parameters in the DenseNet model,
        as well as the number of parameters corresponding to the convolutional
        and fully-connected (classifier) parts of the model.

        Returns:
            total_parameters (int) - total number of parameters in the model.
            conv_params (int) - number of parameters belonging to the
                convolutional part of the model.
            fc_params (int) - number of parameters belonging to the
                fully-connected part of the model (i.e., the classifier).
        """
        total_parameters = 0
        conv_params = 0
        fc_params = 0

        for name, param in self.named_parameters():
            total_parameters += param.numel()
            if 'classifier' in name:
                fc_params += param.numel()
            else:
                conv_params += param.numel()

        return total_parameters, conv_params, fc_params
