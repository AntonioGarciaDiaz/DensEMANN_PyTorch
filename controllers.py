# DensEMANN controller code, for PyTorch + fastai DensEMANN implementation.
# Based on:
# - TensorFlow DensEMANN implementation by me:
#   https://github.com/AntonioGarciaDiaz/Self-constructing_DenseNet
# - PyTorch efficient DenseNet-BC implementation by Geoff Pleiss:
#   https://github.com/gpleiss/efficient_densenet_pytorch
# - TensorFlow DenseNet implementation by Illarion Khlestov:
#   https://github.com/ikhlestov/vision_networks

import numpy as np
import os
import pandas as pd
import tempfile
import time
import torch
from collections import deque
from datetime import timedelta, datetime
from fastai.vision.all import *
from fasterai.sparse.all import *
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

from callbacks import *
from datasets import *
from modded_callbacks import *
from models import DenseNet
from schedules import sched_dsd, sched_dsd_original
from cutout import Cutout


class DensEMANN_controller(object):
    """
    Controller for training and/or testing a DenseNet architecture, optionnaly
    using DensEMANN to self-construct it.
    Mostly based on the efficient DenseNet demo by Geoff Pleiss, with some
    functionalities ported from my TensorFlow DensEMANN implementation
    (itself based on Illarion Khlestov's DenseNet implementation).

    Args:
        train (bool) - whether or not to train the model (default True).
        test (bool) - whether or not to test the model (default True).
        source_experiment_id (str or None) - an experiment ID for an existing
            source model (default None, i.e. create a new model).
        import_weights (bool) - whether or not to import the model's weights
            and trainable values as well as its architecture / hyperparameters
            (default True).
        reuse_files (bool) - whether to reuse the files of the source model
            (feature logs, model and hypers files) or create new ones
            (default False, i.e. create new files).

        data (str or None) - path to directory where data should be loaded
            from/downloaded (default None, i.e. a new folder at the tempfile).
        load (str or None) - path to directory where a model should optionally
            be loaded from (default None, i.e. a 'ft-logs' folder in the
            current working directory).
        save (str or None) - path to directory where the model and ft-logs
            should be saved to (default None, i.e. a 'ft-logs' folder in the
            current working directory).
        growth_rate (int) - number of features added per DenseNet layer,
            corresponding to the number of convolution filters in that layer
            (default 12).
        layer_num_list (str) - the block configuration in string form:
            the number of convolution layers in each block, separated by commas
            (default '1').
        filter_num_list (str or None) - the block configuration with an extra
            level of detail: the number of convolution filters in each layer
            (separated by commas) in each block (separated by dot commas)
            (default None, i.e. unused, overrides layer_num_list if not None).
        update_growth_rate (bool) - whether or not to update the DenseNet's
            default growth rate value before each layer or block addition,
            using the previous layer's final number of filters as the new value
            (default True).
        keep_prob (float) - keep probability for dropout, if keep_prob == 1
                dropout will be disabled (default 1.0, i.e. disabled).
        model_type (str) - model type name ('DenseNet' or 'DenseNet-BC',
            default 'DenseNet-BC').
        dataset (str) - dataset name, if followed by a '+' data augmentation
            is added (default 'C10+', i.e. CIFAR-10 with data augmentation).
        resize (int or None) - optional size to resize the dataset's images.
            N.B.: If 32 or lower, small_inputs applies.
        cutout (float) - whether or not to apply CutOut data augmentation.
        reduction (float) - reduction (theta) at transition layers for
                DenseNets with compression (DenseNet-BC) (default 0.5).

        efficient (bool) - whether or not to use the implementation with
            checkpointing (slow but memory efficient) (default False).
        train_size (int) - size of training set (default 45000).
        valid_size (int) - size of validation set (default 5000).
        n_epochs (int) - number of epochs for training, when not
            self-constructing or in DensEMANN variants 0 and 1 (default 300).
        lim_n_epochs (int) - upper limit to the number of training epochs when
            self-constructing (default 99999).
        batch_size (int) - number of images in a training batch (default 64).
        lr (float) - initial learning rate (default 0.1).
        gamma (float) - multiplicative factor for scheduled LR modifications
            (default 0.1, i.e. a division by 10).
        rlr_1 (float) - first scheduling milestone for multiplying the LR by
            gamma when following DensEMANN's LR modification schedule
            (default 0.5, i.e. 50% through the training process).
        rlr_2 (float) - second scheduling milestone for multiplying the LR by
            gamma when following DensEMANN's LR modification schedule
            (default 0.75, i.e. 75% through the training process).
        wd (float) - weight decay for the loss function (default 0.0001).
        momentum (float) - momentum for the optimizer (default 0.9).
        seed (int or None) - optional seed for the random number generator
            (default None).

        should_self_construct (bool) - whether or not to use the DensEMANN
            self-constructive algorithm (default True).
        should_sparsify (bool) - if not using DensEMANN, whether or not to run
            a sparsification schedule on the network (default True).
        should_change_lr (bool) - whether or not the learning rate will be
            modified during training (default True).

        self_constructing_var (int) - DensEMANN variant to use, if the int does
            not identify any variant the most recent variant is used
            (default -1, i.e. most recent variant).
        self_constr_rlr (int) - learning rate reduction variant to use with
            DensEMANN, if the int does not identify any variant then variant 0
            is used (default 0).

        block_count (int) - minimum number of blocks in the network for the
            self-construction process to end (default 1).
        new_block_mode (str) - specifies how to add new blocks to the network:
            'from_scratch', 'brutal_copy', 'incremental_copy' or 'reset_copy'
            (default 'from_scratch').
        layer_cs (str) - 'layer CS', preferred interpretation of CS values
            when evaluating layers (using 'relevance' or 'spread')
            (default relevance).
        asc_thresh (int) - ascension threshold for self-constructing
            (default 10).
        patience_param (int) - patience parameter for self-constructing
            at layer level (default 200).
        std_tolerance (int) - St.D. tolerance for layer-level self-constructing
            (default 0.1).
        std_window (int) - St.D. window for layer-level self-constructing
            (default 50).
        impr_thresh (int) - improvement threshold for layer-level
            self-constructing (default 0.01).
        preserve_transition (bool) - whether or not to preserve the transition
            to classes after layer additions (default True).
        remove_last_layer (bool) - whether or not to undo the last layer's
            addition after building a dense block (default True).

        expansion_rate (int) - rate at which new convolutions are added
            together during the self-construction of a dense layer (default 1).
        dkCS_smoothing (int) - memory window for each filter's kCS during
            filter-level self-constructing (to smooth out the calculation of
            the kCS derivate) (default 10).
        dkCS_std_window (int) - St.D. window for each filter's kCS derivate
            during filter-level self-constructing (default 30).
        dkCS_stl_thresh (float) - settling threshold for each filter's kCS
            derivate during filter-level self-constructing (default 0.001).
        auto_usefulness_thresh (float) - usefulness threshold for filters as
            a fraction between 0 and 1 (used for automatically calculating the
            actual usefulness threshold) (default 0.8).
        auto_uselessness_thresh (float) - uselessness threshold for filters as
            a fraction between 0 and 1 (used for automatically calculating the
            actual uselessness threshold) (default 0.2).
        m_asc_thresh (int) - micro-ascension threshold for self-constructing
            at filter-level (default 5).
        m_patience_param (int) - micro-patience parameter for self-constructing
            at filter-level (default 300).
        complementarity (bool) - whether or not to use complementarity when
            adding new filters during filter-level self-constructing.
        dont_prune_beyond (int) - minimum number of filters that should remain
            after a pruning operation (default 1).
        m_re_patience_param (int) - alternate micro-patience parameter for the
            micro-recovery stage (default 1000).

        end_sparsity (float) - percentage of the trainable parameters that
            should be zeroed-out during scheduled sparsification (default 50).
        spars_granularity (str) - Granularity for sparsification: 'weight',
            'shared_weight', 'column', 'row', 'channel', 'kernel' or 'filter'
            (default 'filter').
        spars_method (str) - sparsification method : either 'local' or 'global'
            (default 'global').
        spars_sched_func (str) - schedule function for the sparsifier:
            'one_shot', 'iterative', 'sched_agp', 'sched_onecycle' or
            'sched_dsd' (default 'sched_dsd').
        spars_start_epoch (int) - training epoch at which sparsification starts
            (default 0).
        spars_end_epoch (int or None) - training epoch at which sparsification
            ends (default None, i.e. the end of the training).
        lth (bool) - whether or not to perform 'Lottery Ticket Hypothesis'-
            style rewinding after every pruning operation during sparsification
            (default False).
        lth_rewind_epoch (int) - reference training epoch for LTH-style
            rewinding (default 0).
        spars_iterative_n_steps (int) - number of steps for the iterative
            sparsifier schedule function (default 3).
        spars_sched_onecycle_alpha (float) - alpha value for the sched_onecycle
            sparsifier schedule function (default 14).
        spars_sched_onecycle_beta (float) - beta value for the sched_onecycle
            sparsifier schedule function (default 6).
        spars_sched_dsd_middle (float or None) - mid-pruning sparsity
            percentage for the sched_dsd schedule function
            (default None, i.e. halfway between end_sparsity and 100).
        spars_sched_dsd_pattern (str) - the pattern for the pruning and
            unpruning in the sched_dsd schedule function (default 'cos').
        spars_sched_dsd_iterations (int) - the number of pruning and
            unpruning in the sched_dsd schedule function (default 'cos').
        spars_sched_dsd_middle_pos (float) - relative position corresponding to
            mid-pruning for the sched_dsd schedule function (default 0.5).

        rlr_start_epoch (int) - training epoch at which the scheduled learning
            rate modification starts, relative to the epoch in which it is
            activated (default 0).
        rlr_end_epoch (int or None) - training epoch at which the scheduled
            learning rate modification ends, relative to the epoch in which it
            is activated (default None, i.e. use the length of a 'patience'
            cycle if using DensEMANN, or otherwise the value of n_epochs).
        prebuilt_rlr (str) - learning rate reduction schedule to use when
            training prebuilt networks (default 'DensEMANN', i.e. the same
            schedule that is used for DensEMANN's learning rate reduction).

        DensEMANN_init (bool) - whether or not to use the DensEMANN
            initialization method for prebuilt DenseNet (default True).

        should_save_model (bool) - whether or not to save the model and
            relevant hyperparameters (default True).
        should_save_ft_logs (bool) - whether or not to save feature logs
            (default True).
        save_model_every_epoch (bool) - whether or not to save the model at
            every epoch (or with every feature log entry), as opposed to only
            when the validation loss decreases or when required by DensEMANN
            (default False).
        every_epoch_until (int or None) - optional number of epochs during
            which the model is saved at every epoch (if not None,
            save_model_every_epoch is considered as True), and after which it
            is only saved when the validation loss decreases or when required
            by DensEMANN (default None, i.e. if save_model_every_epoch is True
            the model is always saved at every epoch).
        keep_intermediary_model_saves (bool) - whether or not to keep
            intermediary model saves from DensEMANN after it is used
            (default False).
        ft_freq (int) - number of epochs between two feature log entries,
            and between two model saves if save_model_every_epoch is True
            (default 1).
        ft_comma (str) - 'comma' separator in the CSV feature logs
            (default ';').
        ft_decimal (str) - 'decimal' separator in the CSV feature logs
            (default ',').
        add_ft_kCS (bool) - whether or not to add the kCS values from filters
            in each layer to the CSV log.

    Attributes:
        should_train (bool) - 'train' from args.
        should_test (bool) - 'test' from args.
        data (str) - from args (or the default folder if None in args).
        load (str) - from args (or the default folder if None in args).
        save (str) - from args (or the default folder if None in args).
        growth_rate (int) - from args.
        block_config (list of int or list of list of int) - the block
            configuration in list form: a list containing the number of
            convolution layers in each block, and optionally the number of
            convolution filters in each layer (taken from the 'filter_num_list'
            arg if it exists, else from the 'layer_num_list' arg).
        keep_prob (float) - from args.
        bc_mode (bool) - model type, True if DenseNet-BC, False if DenseNet
            (taken from the 'model_type' arg).
        dataset (str) - from args.
        reduction (float) - from args.

        efficient (bool) - from args.
        n_epochs (int) - from args.
        lim_n_epochs (int) - from args.
        batch_size (int) - from args.
        lr (float) - from args.
        gamma (float) - from args.
        rlr_1 (float) - from args.
        rlr_2 (float) - from args.
        wd (float) - from args.
        momentum (float) - from args.
        seed (int or None) - from args.

        should_self_construct (bool) - from args.
        should_sparsify (bool) - from args.
        should_change_lr (bool) - from args.

        self_constructing_var (int) - from args.
        self_constr_rlr (int) - from args.
        has_micro_algo (bool) - whether or not the DensEMANN variant in use
            contains a micro-algorithm (i.e. self-constructs at filter level).

        self_constr_kwargs (dict) - a keyword argument dictionary containing
            block_count, new_block_mode, layer_cs, asc_thresh, patience_param,
            std_tolerance, std_window, impr_thresh, preserve_transition,
            remove_last_layer,
            and optionally also expansion_rate, dkCS_smoothing,
            dkCS_std_window, dkCS_stl_thresh, auto_usefulness_thresh,
            auto_uselessness_thresh, m_asc_thresh, m_patience_param,
            complementarity, dont_prune_beyond, and m_re_patience_param;
            all from args.
        sparsifier_kwargs (dict) - a keyword argument dictionnary containing
            end_sparsity, spars_granularity (as granularity),
            spars_method (as method), the schedule function (as sched_func),
            spars_start_epoch (as start_epoch), spars_end_epoch (as end_epoch),
            lth, and lth_rewind_epoch (as rewind_epoch).

        rlr_start_epoch (int) - from args.
        rlr_end_epoch (int or None) - from args.
        prebuilt_rlr (str) - from args.

        should_save_model (bool) - from args.
        should_save_ft_logs (bool) - from args.
        save_model_every_epoch (bool) - from args, but set to False if not
            using a validation set, else set to True if every_epoch_until is
            not None.
        every_epoch_until (int or None) - from args.
        remove_intermediary_model_saves (bool) - should_save_model and not
            keep_intermediary_model_saves, both from args.
        ft_freq (int) - from args.
        ft_comma (str) - from args.
        ft_decimal (str) - from args.
        add_ft_kCS (bool) - from args.

        source_experiment_id (str or None) - from args.
        experiment_id (str) - a string that identifies the current execution,
            used in file names for the model, feature logs, etc. (format:
            [model_type]_[dataset]_DensEMANN_[var]_[rlr]_[update_k]_[date] for
            DensEMANN, [model_type]_[dataset]_prebuilt_k=[initial_k]_[fnl/lnl]
            _[rlr(+spars)]_[update_k]_[date] for prebuilt).
        num_input_features (int) - number of input feature maps corresponding
            to each example in the dataset.
        num_classes (int) - number of classification classes in the dataset.
        small_inputs (bool) - set to True if images are 32x32 or similar,
            otherwise assumes images are larger (default True).
        random_crop (int) - desired output size of images (in pixels) after
            random crop, when using data augmentation (dataset with '+').
        model (DenseNet) - DenseNet model to be trained.
        train_set (VisionDataset) - training set.
        valid_set (VisionDataset or None) - validation set.
        test_set (VisionDataset) - test set.
    """
    def __init__(self, train=True, test=True,
                 source_experiment_id=None, import_weights=True,
                 reuse_files=False, data=None, load=None, save=None,
                 growth_rate=12, layer_num_list='1', filter_num_list=None,
                 update_growth_rate=True, keep_prob=1.0,
                 model_type='DenseNet-BC', dataset='C10+', resize=None,
                 cutout=False, reduction=0.5,
                 efficient=False, train_size=45000, valid_size=5000,
                 n_epochs=300, lim_n_epochs=99999999,
                 batch_size=64, lr=0.1, gamma=0.1, rlr_1=0.5, rlr_2=0.75,
                 wd=0.0001, momentum=0.9, seed=None,
                 should_self_construct=True, should_sparsify=True,
                 should_change_lr=True,
                 self_constructing_var=-1, self_constr_rlr=0,
                 block_count=1, new_block_mode='from_scratch',
                 layer_cs='relevance', asc_thresh=10, patience_param=200,
                 std_tolerance=0.1, std_window=50, impr_thresh=0.01,
                 preserve_transition=True, remove_last_layer=True,
                 expansion_rate=1, dkCS_smoothing=10, dkCS_std_window=30,
                 dkCS_stl_thresh=0.001, auto_usefulness_thresh=0.8,
                 auto_uselessness_thresh=0.2, m_asc_thresh=5,
                 m_patience_param=300, complementarity=True,
                 dont_prune_beyond=1, m_re_patience_param=1000,
                 end_sparsity=50, spars_granularity='filter',
                 spars_method='global', spars_sched_func='sched_agp',
                 spars_start_epoch=0, spars_end_epoch=None,
                 lth=False, lth_rewind_epoch=0,
                 spars_iterative_n_steps=3, spars_sched_onecycle_alpha=14,
                 spars_sched_onecycle_beta=6, spars_sched_dsd_middle=None,
                 spars_sched_dsd_pattern='cos', spars_sched_dsd_iterations=1,
                 spars_sched_dsd_middle_pos=0.5,
                 rlr_start_epoch=0, rlr_end_epoch=None,
                 prebuilt_rlr='DensEMANN', DensEMANN_init=True,
                 should_save_model=True, should_save_ft_logs=True,
                 save_model_every_epoch=False, every_epoch_until=None,
                 keep_intermediary_model_saves=False,
                 ft_freq=1, ft_comma=';', ft_decimal=',', add_ft_kCS=True):
        """
        Initializer for the DensEMANN_controller class.

        Raises:
            Exception: DensEMANN variant 'self_constructing_var' not yet
                implemented. Implemented DensEMANN variants: [4, 5, 6, 7].
            Exception: dataset 'DATASET' not yet supported.
                Supported datasets: [supported_datasets].
            Exception: 'load' is not a dir.
            Exception: 'save' is not a dir.
            Exception: source hyperparameters for 'source_experiment_id' not
                found at dir 'save'. Please provide a valid experiment ID.
            Exception: source model for 'source_experiment_id' not found at dir
                'save'. Please provide a valid experiment ID.
            Exception: size 'valid_size' was specified for the validation set,
                which is greater than the original training set size for
                this dataset ('len(indices)').
        """
        # List of implemented DensEMANN and reduce LR variants.
        implemented_variants = [4, 5, 6, 7]
        implemented_rlr = [0, 1]
        # Check if the variant is implemented.
        # TODO: Remove these error messages once all variants are implemented.
        if should_self_construct and self_constructing_var >= 0 and (
                self_constructing_var <= implemented_variants[-1]) and (
                self_constructing_var not in implemented_variants):
            raise Exception(('DensEMANN variant \'%d\' not yet implemented. ' +
                            'Implemented DensEMANN variants: %s.') % (
                                self_constructing_var,
                                str(implemented_variants)))
        # Enforce defaults for DensEMANN and reduce LR variants.
        if self_constructing_var >= 1000:
            self_constructing_var = 1000
        elif (self_constructing_var) <= 0 or (
                self_constructing_var >= implemented_variants[-1]):
            self_constructing_var = implemented_variants[-1]
        if self_constr_rlr not in implemented_rlr:
            self_constr_rlr = 0

        # Dictionnary of supported datasets, containing corresponding number of
        # input channels, classes, small_inputs, and end size for random crop.
        supported_datasets = {'C10': [3, 10, True, 32],
                              'C100': [3, 100, True, 32],
                              'SVHN': [3, 10, True, 32],
                              'ImageNet': [3, 1000, False, 224],
                              'FMNIST': [1, 10, True, 28],
                              'FER2013': [1, 7, True, 48]}
        # Check if the dataset is supported.
        augment_data = dataset[-1] == '+'
        dataset_name = dataset[:-1] if augment_data else dataset
        if dataset_name not in supported_datasets:
            raise Exception(('dataset \'%s\' not yet supported. ' +
                            'Supported datasets: %s.') % (
                                dataset_name, str(supported_datasets.keys())))
        # Handle data, load and save directories.
        self.data = data
        self.load = load
        self.save = save
        # Handle cases where these directories are not given or relative.
        if data is None:
            self.data = os.path.join(tempfile.gettempdir(), dataset_name)
        elif data.startswith("./"):
            self.data = os.path.join(os.getcwd(), data[2:], dataset_name)
        else:
            self.data = os.path.join(data, dataset_name)
        if load is None:
            self.load = os.path.join(os.getcwd(), "ft-logs")
        elif load.startswith("./"):
            self.load = os.path.join(os.getcwd(), load[2:])
        if save is None:
            self.save = os.path.join(os.getcwd(), "ft-logs")
        elif save.startswith("./"):
            self.save = os.path.join(os.getcwd(), save[2:])
        # Create the save directory for the model (if it doesn't exist yet).
        if not os.path.exists(self.save):
            os.makedirs(self.save)
        # Check if the load and save directories exist.
        if source_experiment_id is not None and not os.path.isdir(self.load):
            raise Exception('%s is not a dir.' % self.load)
        if not os.path.isdir(self.save):
            raise Exception('%s is not a dir.' % self.save)

        # Copy (and/or adapt) interesting args as class attributes.
        self.should_train = train
        self.should_test = test
        self.growth_rate = growth_rate
        # Get densenet configuration from layer_num_list or filter_num_list.
        if filter_num_list:
            self.block_config = list(map(str, filter_num_list.split(';')))
            for i in range(len(self.block_config)):
                self.block_config[i] = list(
                    map(int, self.block_config[i].split(',')))
        else:
            self.block_config = list(map(int, layer_num_list.split(',')))
        self.update_growth_rate = update_growth_rate
        self.keep_prob = keep_prob
        # Get bc_mode attribute from model_type arg.
        self.bc_mode = model_type != 'DenseNet'
        self.dataset = dataset
        self.reduction = reduction
        self.efficient = efficient
        self.n_epochs = n_epochs
        self.lim_n_epochs = lim_n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.rlr_1 = rlr_1
        self.rlr_2 = rlr_2
        self.wd = wd
        self.momentum = momentum
        self.seed = seed
        self.should_self_construct = should_self_construct
        self.should_sparsify = should_sparsify
        self.should_change_lr = should_change_lr
        self.self_constructing_var = self_constructing_var
        self.self_constr_rlr = self_constr_rlr
        self.has_micro_algo = should_self_construct and (
            self_constructing_var < 0 or self_constructing_var >= 4)
        if self.should_self_construct:
            self.self_constr_kwargs = {
                "block_count": block_count, "new_block_mode": new_block_mode,
                "layer_cs": layer_cs,
                "asc_thresh": asc_thresh, "patience_param": patience_param,
                "std_tolerance": std_tolerance, "std_window": std_window,
                "impr_thresh": impr_thresh,
                "preserve_transition": preserve_transition,
                "remove_last_layer": remove_last_layer}
            if self.has_micro_algo:
                self.self_constr_kwargs.update({
                    "expansion_rate": expansion_rate,
                    "dkCS_smoothing": dkCS_smoothing,
                    "dkCS_std_window": dkCS_std_window,
                    "dkCS_stl_thresh": dkCS_stl_thresh,
                    "auto_usefulness_thresh": auto_usefulness_thresh,
                    "auto_uselessness_thresh": auto_uselessness_thresh,
                    "m_asc_thresh": m_asc_thresh,
                    "m_patience_param": m_patience_param,
                    "complementarity": complementarity,
                    "dont_prune_beyond": dont_prune_beyond,
                    "m_re_patience_param": m_re_patience_param})
        elif self.should_sparsify:
            # Build the sparsifier schedule function to use.
            true_spars_sched_func = eval(spars_sched_func)
            if spars_sched_func == "iterative":
                true_spars_sched_func = lambda start, end, pos: iterative(
                    start, end, pos, n_steps=spars_iterative_n_steps)
            elif spars_sched_func == 'sched_onecycle':
                true_spars_sched_func = lambda start, end, pos: sched_onecycle(
                    start, end, pos, α=spars_sched_onecycle_alpha,
                    β=spars_sched_onecycle_beta)
            elif spars_sched_func == 'sched_dsd':
                true_spars_sched_func = lambda start, end, pos: sched_dsd(
                    start, end, pos, middle=spars_sched_dsd_middle,
                    pattern=spars_sched_dsd_pattern,
                    iterations=spars_sched_dsd_iterations,
                    middle_pos=spars_sched_dsd_middle_pos)
            # elif spars_sched_func == 'sched_dsd_original':
            #     true_spars_sched_func = lambda start, end, pos:
            #         sched_dsd_original(start, end, pos,
            #                            middle=spars_sched_dsd_middle)
            self.sparsifier_kwargs = {
                "end_sparsity": end_sparsity, "granularity": spars_granularity,
                "method": spars_method, "sched_func": true_spars_sched_func,
                "start_epoch": spars_start_epoch, "end_epoch": spars_end_epoch,
                "lth": lth, "rewind_epoch": lth_rewind_epoch}
        self.rlr_start_epoch = rlr_start_epoch
        self.rlr_end_epoch = rlr_end_epoch
        self.prebuilt_rlr = prebuilt_rlr
        self.should_save_model = should_save_model
        self.should_save_ft_logs = should_save_ft_logs
        self.save_model_every_epoch = bool(valid_size and (
            save_model_every_epoch or every_epoch_until is not None))
        self.every_epoch_until = every_epoch_until
        self.remove_intermediary_model_saves = (
            should_save_model and not keep_intermediary_model_saves)
        self.ft_freq = ft_freq
        self.ft_comma = ft_comma
        self.ft_decimal = ft_decimal
        self.add_ft_kCS = add_ft_kCS

        # If source ID is specified.
        self.source_experiment_id = source_experiment_id
        if source_experiment_id is not None:
            # Check if the hypers file exists.
            if not os.path.isfile(os.path.join(
                    self.load, source_experiment_id + '_hypers.pkl')):
                raise Exception(
                    ('source hyperparameters for %s not found at dir %s.'
                     + ' Please provide a valid experiment ID.')
                    % (source_experiment_id + '_hypers.pkl', self.load))
            # Also, if source model should be used, check if it exists.
            if import_weights is not None and not os.path.isfile(os.path.join(
                    self.load, source_experiment_id + '_model.pth')):
                raise Exception(
                    ('source model for %s not found at dir %s.'
                     + ' Please provide a valid experiment ID.')
                    % (source_experiment_id + '_model.pth', self.load))
            # Copy some hyperparameters from source file.
            with open(os.path.join(
                    self.load, source_experiment_id + '_hypers.pkl'), 'rb'
                    ) as file:
                values_dict = pickle.load(file)
                self.growth_rate = values_dict["growth_rate"]
                self.block_config = values_dict["block_config"]
                self.update_growth_rate = values_dict["update_growth_rate"]
                self.bc_mode = values_dict["bc_mode"]
                self.reduction = values_dict["reduction"]
        # Set identifier for the experiment (depends on source model).
        if reuse_files and source_experiment_id is not None:
            self.experiment_id = source_experiment_id
            # In this case, the save and load directories should be the same.
            self.save = self.load
            # Also specify in the ft-logs that the model was reloaded.
            if self.should_save_ft_logs:
                with open(os.path.join(
                        self.save, self.experiment_id + '_ft_log.csv'),
                        'a') as f:
                    f.write(('\nReloaded in {}\n\n').format(
                             datetime.now().strftime("%Y_%m_%d_%H%M%S")))
                    if should_self_construct:
                        f.write(('Using DensEMANN var #{} with rlr #{}\n\n'
                                 ).format(self.self_constructing_var,
                                          self.self_constr_rlr))
        else:
            if should_self_construct:
                self.experiment_id = '%s_%s_DensEMANN_var%d_%s_%s_%s' % (
                    "DenseNet-BC" if self.bc_mode else "DenseNet",
                    self.dataset, self.self_constructing_var,
                    ("rlr%d" % self.self_constr_rlr) if self.should_change_lr
                    else "NOrlr", "update-k" if self.update_growth_rate
                    else "same-k", datetime.now().strftime("%Y_%m_%d_%H%M%S"))
            else:
                self.experiment_id = '%s_%s_prebuilt_k=%d_%s=%s_%s%s_%s_%s' % (
                    "DenseNet-BC" if self.bc_mode else "DenseNet",
                    self.dataset, self.growth_rate,
                    "fnl" if type(self.block_config[0]) is list else "lnl",
                    ";".join(",".join(str(k) for k in self.block_config[layer])
                             for layer in range(len(self.block_config)))
                    if type(self.block_config[0]) is list
                    else ",".join(str(layer) for layer in self.block_config),
                    ("rlr=%s" % self.prebuilt_rlr) if self.should_change_lr
                    else "NOrlr",
                    ("_spars={}({}%){}".format(
                        spars_sched_func, end_sparsity, "+lth" if lth else ""))
                    if self.should_sparsify else "",
                    "update-k" if self.update_growth_rate else "same-k",
                    datetime.now().strftime("%Y_%m_%d_%H%M%S"))
            # If using a source model, specify that in the ft-logs.
            if source_experiment_id is not None and self.should_save_ft_logs:
                with open(os.path.join(
                        self.save, self.experiment_id + '_ft_log.csv'),
                        'a') as f:
                    f.write(('Using source model \"{}\"\n\n').format(
                             source_experiment_id))

        # Number of input features, classes, and small_inputs for the dataset.
        self.num_input_features = supported_datasets[dataset_name][0]
        self.num_classes = supported_datasets[dataset_name][1]
        self.small_inputs = supported_datasets[dataset_name][2]
        self.random_crop = supported_datasets[dataset_name][3]
        if resize:
            self.small_inputs = resize <= 32
            self.random_crop = resize

        # Get the right normalize transform for each dataset.
        if dataset_name in ["C10", "C100"]:
            normalize_tfm = transforms.Normalize(*cifar_stats)
        elif dataset_name == "SVHN":
            normalize_tfm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        elif dataset_name in ["FMNIST", "FER2013"]:
            normalize_tfm = transforms.Normalize(mean=[0.5], std=[0.5])
        # elif dataset_name == "MNIST"
        #     normalize_tfm = transforms.Normalize(*mnist_stats)
        else:
            # Imagenet stats should be good for most problems and datasets.
            normalize_tfm = transforms.Normalize(*imagenet_stats)

        # Define data transforms for resizing.
        resize_transforms = None
        # Imagenet always needs resizing and center crop.
        if dataset_name == "ImageNet":
            resize_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])
            # If further resizing is specified.
            if resize is not None:
                # If that resize is not already in a dir, apply it.
                if not os.path.isdir(os.path.join(
                        self.data,
                        "ILSVRC/Data/CLS-LOC/resized_{}x".format(resize))):
                    resize_transforms = transforms.Compose(
                        resize_transforms.transforms
                        + [transforms.Resize(resize)])
                # Else, erase the resize transforms (already in the dir).
                else:
                    resize_transforms = None
        # Otherwise apply resizing if specified.
        elif resize is not None:
            resize_transforms = transforms.Compose([transforms.Resize(resize)])

        # Define data transforms for training and testing.
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize_tfm,
        ])
        # Remove data augmentation transforms if unused.
        if not augment_data:
            train_transforms = test_transforms
        else:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(
                    self.random_crop, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_tfm,
            ])
        # Optionally add a CutOut data augmentation transform.
        if cutout:
            train_transforms = transforms.Compose(
                train_transforms.transforms + [Cutout(n_holes=1, length=16)])
        # Optionally add resize transforms.
        if resize_transforms:
            test_transforms = transforms.Compose(
                resize_transforms.transforms + test_transforms.transforms)
            train_transforms = transforms.Compose(
                resize_transforms.transforms + train_transforms.transforms)

        # print(resize_transforms)
        # print(test_transforms)
        # print(train_transforms)

        # Define training set and testing set (select the right dataset).
        if dataset_name == "C10":
            self.train_set = datasets.CIFAR10(
                self.data, train=True, transform=train_transforms,
                download=not os.path.isdir(self.data))
            self.test_set = datasets.CIFAR10(
                self.data, train=False, transform=test_transforms,
                download=False)
        elif dataset_name == "C100":
            self.train_set = datasets.CIFAR100(
                self.data, train=True, transform=train_transforms,
                download=not os.path.isdir(self.data))
            self.test_set = datasets.CIFAR100(
                self.data, train=False, transform=test_transforms,
                download=False)
        elif dataset_name == "SVHN":
            should_download = not os.path.isdir(self.data)
            self.train_set = ConcatDataset([
                datasets.SVHN(
                    self.data, split='train', transform=train_transforms,
                    download=should_download),
                datasets.SVHN(
                    self.data, split='extra', transform=train_transforms,
                    download=should_download)])
            self.test_set = datasets.SVHN(
                self.data, split='test', transform=test_transforms,
                download=should_download)
        elif dataset_name == "ImageNet":
            self.train_set = ImageNetKaggle(
                self.data, 'train', transform=train_transforms,
                resize=resize)
            self.test_set = ImageNetKaggle(
                self.data, 'val', transform=test_transforms,
                resize=resize)
        elif dataset_name == "FMNIST":
            self.train_set = datasets.FashionMNIST(
                self.data, train=True, transform=train_transforms,
                download=not os.path.isdir(self.data))
            self.test_set = datasets.FashionMNIST(
                self.data, train=False, transform=test_transforms,
                download=False)
        elif dataset_name == "FER2013":
            # FER-2013 cannot be downloaded normally.
            # The corresponding CSV file must be at self.data.
            self.train_set = ConcatDataset([
                FER2013(self.data, split='Training',
                        transform=train_transforms),
                FER2013(self.data, split='PrivateTest',
                        transform=train_transforms)])
            self.test_set = FER2013(self.data, split='PublicTest',
                                    transform=test_transforms)

        # If specified, define validation set (cut off training set).
        if train_size or valid_size:
            indices = torch.randperm(len(self.train_set))
            if valid_size:
                # If the valid size exceeds len(indices), an error is raised.
                if valid_size > len(indices):
                    raise Exception(
                        'size %d was specified for the validation set, which'
                        ' is greater than the original training set size for'
                        ' this dataset (%d).' % (valid_size, len(indices)))
                valid_indices = indices[len(indices) - valid_size:]
                self.valid_set = Subset(self.train_set, valid_indices)
            if train_size:
                # If the train size exceeds its max value, it goes back to it.
                max_train_size = (
                    len(indices) - valid_size if valid_size else len(indices))
                if train_size > max_train_size:
                    train_size = max_train_size
                train_indices = indices[:train_size]
                self.train_set = Subset(self.train_set, train_indices)
        if not valid_size:
            self.valid_set = None

        # Define the model to train.
        self.model = DenseNet(
            growth_rate=self.growth_rate,
            block_config=self.block_config,
            DensEMANN_init=(self.should_self_construct or DensEMANN_init),
            update_growth_rate=self.update_growth_rate,
            bc_mode=self.bc_mode,
            reduction=self.reduction,
            num_input_features=self.num_input_features,
            num_init_features=self.growth_rate*2,
            drop_rate=float(1-self.keep_prob),
            num_classes=self.num_classes,
            small_inputs=self.small_inputs,
            efficient=self.efficient,
            seed=self.seed
        )
        print(self.model)
        # Load model state from existing source file if specified.
        if source_experiment_id is not None and import_weights:
            self.model.load_state_dict(torch.load(os.path.join(
                self.load, source_experiment_id + '_model.pth')), strict=False)

        # Calculate and print the total number of parameters in the model.
        num_params = sum(p.numel() for p in self.model.parameters())
        print("Total parameters: ", num_params)

    def run(self):
        """
        Execution routine for the DenseNet model.
        Contains training and testing.
        """
        # PREPARING THE MODEL (PYTORCH) ---------------------------------------
        # ---------------------------------------------------------------------

        # Create a DataLoader for each set (training, validation, test).
        train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            pin_memory=(torch.cuda.is_available()), num_workers=0)
        test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            pin_memory=(torch.cuda.is_available()), num_workers=0)
        if self.valid_set is None:
            valid_loader = None
        else:
            valid_loader = DataLoader(
                self.valid_set, batch_size=self.batch_size, shuffle=True,
                pin_memory=(torch.cuda.is_available()), num_workers=0)

        # Move all model parameters and buffers to GPU (if GPU is available).
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # Wrap model for multi-GPUs, if available and necessary.
        # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        #     print("Using CUDA with device count = {}".format(
        #         torch.cuda.device_count()))
        #     self.model = torch.nn.DataParallel(self.model).cuda()

        # PREPARING THE MODEL (FASTAI) ----------------------------------------
        # ---------------------------------------------------------------------

        # Create the fastai DataLoaders.
        dls = DataLoaders(train_loader,
                          valid_loader if valid_loader else test_loader)

        # Create the optimizer function
        # (stochastic gradient descent with Nesterov momentum).
        def opt_func(params, **kwargs): return OptimWrapper(
            params, torch.optim.SGD, lr=self.lr, momentum=self.momentum,
            nesterov=True, weight_decay=self.wd)

        # Create the callbacks.
        cbs = []
        # If self constructing, create the self-constructing callback.
        if self.should_self_construct:
            cbs.append(DensEMANNCallback(
                self_constructing_var=self.self_constructing_var,
                should_change_lr=self.should_change_lr,
                should_save_model=self.should_save_model,
                should_save_ft_logs=self.should_save_ft_logs,
                **self.self_constr_kwargs))
        # If sparsifying the network, create the callback to do that.
        # N.B.: Sparsification is disabled for DensEMANN, 'elif' ensures this.
        elif self.should_sparsify:
            cbs.append(SparsifyCallback(
                criteria=large_final, **self.sparsifier_kwargs))
        # If scheduling LR changes, create the callback to do that.
        if self.should_change_lr:
            if self.should_self_construct or self.prebuilt_rlr == "DensEMANN":
                # The LR schedule length depends on whether or not DensEMANN is
                # used, and on the DensEMANN variant that is used.
                schedule_length = (
                    self.n_epochs if self.rlr_end_epoch is None
                    else self.rlr_end_epoch) - self.rlr_start_epoch
                if self.has_micro_algo:
                    schedule_length = self.self_constr_kwargs[
                        "m_patience_param"]
                elif self.should_self_construct and (
                        self.self_constructing_var >= 2):
                    schedule_length = self.self_constr_kwargs["patience_param"]
                cbs.append(ReduceLRCallback(
                    rlr_var=(self.self_constr_rlr if self.should_self_construct
                             else 0),
                    lr=self.lr, gamma=self.gamma,
                    rlr_1=self.rlr_1, rlr_2=self.rlr_2,
                    first_epoch=self.rlr_start_epoch,
                    schedule_length=schedule_length,
                    self_construct_mode=self.should_self_construct))
            else:
                # In any case, the final LR value is the same as for DensEMANN
                final_lr = self.lr*self.gamma*self.gamma
                sched = {'lr': None}
                if self.prebuilt_rlr == 'lin':
                    rlr_schedule = SchedLin(self.lr, final_lr)
                elif self.prebuilt_rlr == 'cos':
                    rlr_schedule = SchedCos(self.lr, final_lr)
                elif self.prebuilt_rlr == 'exp':
                    rlr_schedule = SchedExp(self.lr, final_lr)
                elif self.prebuilt_rlr == 'poly':
                    rlr_schedule = SchedPoly(self.lr, final_lr, 0.5)
                # In case the schedule has got beginning and end epochs.
                if self.rlr_start_epoch != 0 or self.rlr_end_epoch is not None:
                    combine_lists = [[1], [rlr_schedule]]
                    if self.rlr_end_epoch is not None:
                        combine_lists[0] = [self.rlr_end_epoch/self.n_epochs
                                            ] + combine_lists[0]
                        combine_lists[0][1] -= combine_lists[0][0]
                        combine_lists[1].append(SchedNo(final_lr, final_lr))
                    if self.rlr_start_epoch != 0:
                        combine_lists[0] = [self.rlr_start_epoch/self.n_epochs
                                            ] + combine_lists[0]
                        combine_lists[0][1] -= combine_lists[0][0]
                        combine_lists[1] = [SchedNo(self.lr, self.lr)
                                            ] + combine_lists[1]
                    sched['lr'] = combine_scheds(*combine_lists)
                else:
                    sched['lr'] = rlr_schedule
                cbs.append(ParamScheduler(sched))
        # If saving the model, create the callback that saves the best one yet,
        # and the callback that saves relevant hyperparameters.
        if self.should_save_model:
            cbs.append(CustomSaveModelCallback(fname=os.path.join(
                self.save, self.experiment_id + '_model'),
                every_epoch=self.ft_freq*int(self.save_model_every_epoch),
                every_epoch_until=self.every_epoch_until))
            cbs.append(SaveHypersCallback(fname=os.path.join(
                self.save, self.experiment_id + '_hypers.pkl'),
                source_fname=os.path.join(
                    self.load, self.source_experiment_id + '_hypers.pkl')
                if self.source_experiment_id is not None else None))
        # If saving ft-logs, create the callback to write the ft-log file.
        if self.should_save_ft_logs:
            cbs.append(CustomCSVLogger(fname=os.path.join(
                self.save, self.experiment_id + '_ft_log.csv'),
                add_ft_kCS=self.add_ft_kCS, ft_freq=self.ft_freq,
                ft_comma=self.ft_comma, ft_decimal=self.ft_decimal))

        # Create the Learner from the model, optimizer and callbacks.
        learn = Learner(dls, self.model, lr=self.lr,
                        loss_func=nn.CrossEntropyLoss(),
                        opt_func=opt_func, metrics=[
                            accuracy])
        # F1Score(average=None), Precision(average=None), Recall(average=None)])
        # RocAuc(average=None, multi_class='ovo')])

        # TRAINING THE MODEL --------------------------------------------------
        # ---------------------------------------------------------------------

        if self.should_train:
            # Begin counting total training time.
            train_start_time = [time.perf_counter(), time.process_time()]
            if torch.cuda.is_available():
                print("CUDA is available (running on device {}/{})!".format(
                    torch.cuda.current_device()+1, torch.cuda.device_count()
                ))
                train_start_GPU = torch.cuda.Event(enable_timing=True)
                train_end_GPU = torch.cuda.Event(enable_timing=True)
                train_start_GPU.record()
            # Training loop.
            learn.fit(self.lim_n_epochs if self.should_self_construct and (
                self.self_constructing_var >= 2) else self.n_epochs, cbs=cbs)
            # Measure total training time.
            train_time = [time.perf_counter() - train_start_time[0],
                          time.process_time() - train_start_time[1],
                          0]  # Wall, CPU, and GPU time (if using CUDA, else 0)
            if torch.cuda.is_available():
                train_end_GPU.record()
                torch.cuda.synchronize()
                train_time[-1] = train_start_GPU.elapsed_time(train_end_GPU)
            print("\nTRAINING COMPLETE!\n")
            print("TOTAL TRAINING TIME:\n- WALL:\t{}\n- CPU:\t{}\n{}".format(
                str(timedelta(seconds=train_time[0])),
                str(timedelta(seconds=train_time[1])),
                "- GPU:\t{}\n".format(str(timedelta(
                    milliseconds=train_time[-1]))) if train_time[-1] else ""))
            # If DensEMANN was used, print the final architecture.
            if self.should_self_construct:
                total_par, cv_par, fc_par = self.model.count_trainable_params()
                print("Total trainable params: %.1fk" % (total_par / 1e3))
                print("\tConvolutional: %.1fk" % (cv_par / 1e3))
                print("\tFully Connected: %.1fk" % (fc_par / 1e3))
                print("FINAL ARCHITECTURE:\n{}\n".format(self.model))
                # Delete any existing intermediary model files (if specified).
                if self.remove_intermediary_model_saves:
                    prepruning_path = os.path.join(
                        self.save,
                        self.experiment_id + '_model_prepruning.pth')
                    last_layer_path = os.path.join(
                        self.save,
                        self.experiment_id + '_model_last_layer.pth')
                    if os.path.isfile(prepruning_path):
                        os.remove(prepruning_path)
                    if os.path.isfile(last_layer_path):
                        os.remove(last_layer_path)
            # If sparsification was used, print the parameters in use.
            elif self.should_sparsify:
                total_p, iu_p, cv_p, fc_p = self.model.count_trainable_params(
                    count_in_use=True)
                print("Total trainable params: %.1fk" % (total_p / 1e3))
                print("Out of which in use: %.1fk" % (iu_p / 1e3))
                print("\tConvolutional: %.1fk" % (cv_p / 1e3))
                print("\tFully Connected: %.1fk" % (fc_p / 1e3))

        # TESTING THE MODEL ---------------------------------------------------
        # ---------------------------------------------------------------------

        if self.should_test:
            # Begin counting testing time.
            test_start_time = [time.perf_counter(), time.process_time()]
            if torch.cuda.is_available():
                test_start_GPU = torch.cuda.Event(enable_timing=True)
                test_end_GPU = torch.cuda.Event(enable_timing=True)
                test_start_GPU.record()
            # Test on the test set and get results.
            # preds, y, losses = learn.get_preds(dl=test_loader,with_loss=True)
            res = learn.validate(dl=test_loader, ds_idx=2)
            test_time = [time.perf_counter() - test_start_time[0],
                         time.process_time() - test_start_time[1],
                         0]  # Wall, CPU, and GPU time (if using CUDA, else 0)
            if torch.cuda.is_available():
                test_end_GPU.record()
                torch.cuda.synchronize()
                test_time[-1] = test_start_GPU.elapsed_time(test_end_GPU)
            # test_loss = np.mean(losses.tolist())
            # test_accuracy = float(accuracy(preds, y))
            test_loss = res[0]
            test_accuracy = res[1]
            # test_F1 = str(res[2].tolist()).replace(',', ' ')
            # test_Precision = str(res[3].tolist()).replace(',', ' ')
            # test_Recall = str(res[4].tolist()).replace(',', ' ')
            # test_RocAuc = str(res[5].tolist()).replace(',', ' ')
            print("Test:\taccuracy = %f\tloss = %f" % (
                test_accuracy, test_loss))
            # If specified, save relevant data to ft-logs.
            if self.should_save_ft_logs:

                with open(os.path.join(
                        self.save, self.experiment_id + '_ft_log.csv'),
                        'a') as f:
                    to_write = ('test{0}{0}{1:0.5f}{0}{2:0.5f}{0}'
                                # '{3}{0}{4}{0}{5}{0}{6}{0}'
                                ).format(
                                    self.ft_comma, test_loss, test_accuracy)
                    # test_F1, test_Precision, test_Recall, test_RocAuc)
                    f.write(to_write.replace('.', self.ft_decimal))
                    if self.add_ft_kCS:
                        for block in range(len(self.model.block_config)):
                            for layer in range(self.model.block_config[block]):
                                kCS_list = self.model.get_kCS_list_from_layer(
                                    block, layer)
                                f.write(2*self.ft_comma + self.ft_comma.join(
                                    [str(kCS).replace('.', self.ft_decimal)
                                        for kCS in kCS_list]))
                            if block != len(self.model.block_config) - 1:
                                f.write(self.ft_comma)
                    f.write('\n')
            print('Final test accuracy: %.4f' % (test_accuracy))

        # WRITING IMPORTANT INFORMATION AT THE END OF THE FEATURE LOG ---------
        # ---------------------------------------------------------------------

        if self.should_save_ft_logs:
            write_at_end = '\n'
            # Print training and test time.
            if self.should_train:
                write_at_end += 'training wall time{0}{1}\n'.format(
                    self.ft_comma, str(timedelta(seconds=train_time[0])))
                write_at_end += 'training CPU time{0}{1}\n'.format(
                    self.ft_comma, str(timedelta(seconds=train_time[1])))
                if train_time[-1]:
                    write_at_end += 'training GPU time{0}{1}\n'.format(
                        self.ft_comma, str(timedelta(
                            milliseconds=train_time[-1])))
            if self.should_test:
                write_at_end += 'test wall time{0}{1}\n'.format(
                    self.ft_comma, str(timedelta(seconds=test_time[0])))
                write_at_end += 'test CPU time{0}{1}\n'.format(
                    self.ft_comma, str(timedelta(seconds=test_time[1])))
                if test_time[-1]:
                    write_at_end += 'test GPU time{0}{1}\n'.format(
                        self.ft_comma, str(timedelta(
                            milliseconds=test_time[-1])))
            # Count the total, convolutional and F.C. trainable parameters.
            if self.should_sparsify:
                write_at_end += (
                    'total trainable parameters{0}{1}\n' +
                    'out of which in use{0}{2}\n' +
                    'convolutional (in use){0}{3}\n' +
                    'fully connected (in use){0}{4}\n').format(
                        self.ft_comma, *self.model.count_trainable_params(
                            count_in_use=True))
            else:
                write_at_end += (
                    'total trainable parameters{0}{1}\n' +
                    'convolutional{0}{2}\nfully connected{0}{3}\n').format(
                        self.ft_comma, *self.model.count_trainable_params())
            # If DensEMANN was used, print the final architecture.
            if self.should_self_construct:
                write_at_end += '\n\"FINAL ARCHITECTURE:\"\n\"{}\"\n'.format(
                    str(self.model).replace('\n', '\"\n\"'))
            with open(os.path.join(
                    self.save, self.experiment_id + '_ft_log.csv'),
                    'a') as f:
                f.write(write_at_end.replace('.', self.ft_decimal))
