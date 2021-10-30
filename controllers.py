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
import tempfile
import time
import torch
from collections import deque
from datetime import timedelta, datetime
from fastai.vision.all import *
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from callbacks import DensEMANNCallback, ReduceLRCallback, CSVLoggerCustom
from models import DenseNet


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Copied from:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DensEMANN_controller(object):
    """
    Controller for training and/or testing a DenseNet architecture, optionnaly
    using DensEMANN to self-construct it.
    Mostly based on the efficient DenseNet demo by Geoff Pleiss, with some
    functionalities ported from my TensorFlow DensEMANN implementation
    (itself based on Illarion Khlestov's DenseNet implementation)

    Args:
        train (bool) - whether or not to train the model (default True).
        test (bool) - whether or not to test the model (default True).
        source_experiment_id (str or None) - an experiment ID for an existing
            source model (default None, i.e. create a new model).
        reuse_files (bool) - whether to reuse the files of the source model
            (feature logs and model file) or create new ones.

        data (str or None) - path to directory where data should be loaded
            from/downloaded (default None, i.e. a new folder at the tempfile).
        save (str or None) - path to directory where the model and ft-logs
            should be saved to (default None, i.e. a 'ft-logs' folder in the
            current working directory).
        growth_rate (int) - number of features added per DenseNet layer,
            corresponding to the number of convolution filters in that layer
            (default 12).
        layer_num_list (str) - the block configuration in string form:
            the number of convolution layers in each block, separated by commas
            (default '1').
        keep_prob (float) - keep probability for dropout, if keep_prob = 1
                dropout will be disabled (default 1.0, i.e. disabled).
        model_type (str) - model type name ('DenseNet' or 'DenseNet-BC',
            default 'DenseNet-BC').
        dataset (str) - dataset name, if followed by a '+' data augmentation
            is added (default 'C10+', i.e. CIFAR-10 with data augmentation).
        reduction (float) - reduction (theta) at transition layers for
                DenseNets with compression (DenseNet-BC) (default 0.5).

        efficient (bool) - whether or not to use the implementation with
            checkpointing (slow but memory efficient) (default False).
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
            gamma (default 0.5, i.e. 50% through the training process).
        rlr_2 (float) - second scheduling milestone for multiplying the LR by
            gamma (default 0.75, i.e. 75% through the training process).
        wd (float) - weight decay for the loss function (default 0.0001).
        momentum (float) - momentum for the optimizer (default 0.9).
        seed (int or None) - optional seed for the random number generator
            (default None).

        should_self_construct (bool) - whether or not to use the DensEMANN
            self-constructive algorithm (default True).
        should_change_lr (bool) - whether or not the learning rate will be
            modified during training (default True).

        self_constructing_var (int) - DensEMANN variant to be used, if the int
            does not identify any variant the most recent variant is used
            (default -1, i.e. most recent variant).
        self_constr_rlr (int) - learning rate reduction variant to be used
            with DensEMANN, if the int does not identify any variant the most
            recent variant is used (default 0).

        block_count (int) - minimum number of blocks in the network for the
            self-construction process to end (default 1).
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
        acc_smoothing (int) - memory window for network accuracies during
            filter-level self-constructing (to smooth out the calculation of
            the pre-pruning accuracy level) (default 10).

        should_save_model (bool) - whether or not to save the model
            (default True).
        should_save_ft_logs (bool) - whether or not to save feature logs
            (default True).
        ft_freq (int) - number of epochs between two measurements of values
            (e.g. epoch, accuracy, loss, CS) in feature logs (default 1).
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
        save (str) - from args (or the default folder if None in args).
        growth_rate (int) - from args.
        block_config (list of int) - the block configuration in list form:
            a list containing the number of convolution layers in each block
            (taken from the 'layer_num_list' arg).
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
        should_change_lr (bool) - from args.

        self_constructing_var (int) - from args.
        self_constr_rlr (int) - from args.
        has_micro_algo (bool) - whether or not the DensEMANN variant in use
            contains a micro-algorithm (i.e. self-constructs at filter level).

        self_constr_kwargs (dict) - a keyword argument dictionary containing
            block_count, layer_cs, asc_thresh, patience_param, std_tolerance,
            std_window, impr_thresh, preserve_transition, and optionally also
            expansion_rate, dkCS_smoothing, dkCS_std_window, dkCS_stl_thresh,
            auto_usefulness_thresh, auto_uselessness_thresh, m_asc_thresh,
            m_patience_param, complementarity, and acc_smoothing;
            all from args.

        should_save_model (bool) - from args.
        should_save_ft_logs (bool) - from args.
        ft_freq (int) - from args.
        ft_comma (str) - from args.
        ft_decimal (str) - from args.
        add_ft_kCS (bool) - from args.

        experiment_id (str) - a string that identifies the current execution,
            used in file names for the model, feature logs, etc. (format:
            [model_type]_[dataset]_k=[growth_rate]_[layer_num_list]_[date]_).
        num_classes (int) - number of classification classes in the dataset.
        small_inputs (bool) - set to True if images are 32x32, otherwise
            assumes images are larger (default True).
        model (DenseNet) - DenseNet model to be trained.
        train_set (VisionDataset) - training set.
        valid_set (VisionDataset or None) - validation set.
        test_set (VisionDataset) - test set.
    """
    def __init__(self, train=True, test=True,
                 source_experiment_id=None, reuse_files=True,
                 data=None, save=None, growth_rate=12,
                 layer_num_list='1', keep_prob=1.0,
                 model_type='DenseNet-BC', dataset='C10+', reduction=0.5,
                 efficient=False, valid_size=5000, n_epochs=300,
                 lim_n_epochs=99999,
                 batch_size=64, lr=0.1, gamma=0.1, rlr_1=0.5, rlr_2=0.75,
                 wd=0.0001, momentum=0.9, seed=None,
                 should_self_construct=True, should_change_lr=True,
                 self_constructing_var=-1, self_constr_rlr=0,
                 block_count=1, layer_cs='relevance', asc_thresh=10,
                 patience_param=200, std_tolerance=0.1, std_window=50,
                 impr_thresh=0.01, preserve_transition=True,
                 expansion_rate=1, dkCS_smoothing=10, dkCS_std_window=30,
                 dkCS_stl_thresh=0.001, auto_usefulness_thresh=0.8,
                 auto_uselessness_thresh=0.2, m_asc_thresh=5,
                 m_patience_param=300, complementarity=True, acc_smoothing=10,
                 should_save_model=True, should_save_ft_logs=True,
                 ft_freq=1, ft_comma=';', ft_decimal=',', add_ft_kCS=True):
        """
        Initializer for the DensEMANN_controller class.

        Raises:
            Exception: DensEMANN variant 'self_constructing_var' not yet
                implemented. Implemented DensEMANN variants: [4, 5, 6, 7].
            Exception: dataset 'DATASET' not yet supported.
                Supported datasets: [C10, C100, SVHN].
            Exception: 'save' is not a dir.
            Exception: source model source_experiment_id not found at dir save.
                Please provide a valid experiment ID.
        """
        # List of implemented DensEMANN and reduce LR variants.
        implemented_variants = [4, 5, 6, 7]
        implemented_rlr = [0, 1]
        # Check if the variant is implemented.
        # TODO: Remove these error messages once all variants are implemented.
        if self_constructing_var >= 0 and (
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

        # Dictionnary of supported datasets and number of classes.
        supported_datasets = {'C10': 10, 'C100': 100, 'SVHN': 10}
        # Check if the dataset is supported.
        augment_data = dataset[-1] == '+'
        dataset_name = dataset[:-1] if augment_data else dataset
        if dataset_name not in supported_datasets:
            raise Exception(('dataset \'%s\' not yet supported. ' +
                            'Supported datasets: %s.') % (
                                dataset_name, str(supported_datasets.keys())))
        # Handle data and save directories.
        self.data = data
        self.save = save
        # Handle cases where data, save directories are not given or relative.
        if data is None:
            self.data = os.path.join(tempfile.gettempdir(), dataset_name)
        elif data.startswith("./"):
            self.data = os.path.join(os.getcwd(), data[2:])
        if save is None:
            self.save = os.path.join(os.getcwd(), "ft-logs")
        elif save.startswith("./"):
            self.save = os.path.join(os.getcwd(), save[2:])
        # Create the save directory for the model (if it doesn't exist yet).
        if not os.path.exists(self.save):
            os.makedirs(self.save)
        if not os.path.isdir(self.save):
            raise Exception('%s is not a dir.' % self.save)

        # Copy (and/or adapt) interesting args as class attributes.
        self.should_train = train
        self.should_test = test
        self.growth_rate = growth_rate
        # Get densenet configuration from layer_num_list.
        self.block_config = list(map(int, layer_num_list.split(',')))
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
        self.should_change_lr = should_change_lr
        self.self_constructing_var = self_constructing_var
        self.self_constr_rlr = self_constr_rlr
        self.has_micro_algo = should_self_construct and (
            self_constructing_var < 0 or self_constructing_var >= 4)
        if self.should_self_construct:
            self.self_constr_kwargs = {
                "block_count": block_count, "layer_cs": layer_cs,
                "asc_thresh": asc_thresh, "patience_param": patience_param,
                "std_tolerance": std_tolerance, "std_window": std_window,
                "impr_thresh": impr_thresh,
                "preserve_transition": preserve_transition}
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
                    "acc_smoothing": acc_smoothing})
        self.should_save_model = should_save_model
        self.should_save_ft_logs = should_save_ft_logs
        self.ft_freq = ft_freq
        self.ft_comma = ft_comma
        self.ft_decimal = ft_decimal
        self.add_ft_kCS = add_ft_kCS

        # If source model is specified, check if the model file exists.
        if source_experiment_id is not None:
            if not os.path.isfile(os.path.join(
                    self.save, source_experiment_id + '_model.pth')):
                raise Exception(
                    ('source model %s not found at dir %s.'
                     + ' Please provide a valid experiment ID.')
                    % (source_experiment_id + '_model.pth', self.save))
        # Set identifier for the experiment (depends on source model).
        if source_experiment_id is not None and reuse_files:
            self.experiment_id = source_experiment_id
            # Also specify in the ft-logs that the model was reloaded.
            if self.should_save_ft_logs:
                with open(os.path.join(
                        self.save, self.experiment_id + '_ft_log.csv'),
                        'a') as f:
                    f.write(('\nReloaded in {}\n\n').format(
                             datetime.now().strftime("%Y_%m_%d_%H%M%S")))
                    if should_self_construct:
                        f.write(('Using DensEMANN var #{} with rlr #{}\n\n'
                                 ).format(self_constructing_var,
                                          self_constr_rlr))
        else:
            if should_self_construct:
                self.experiment_id = '%s_%s_DensEMANN_var%d_%s_%s' % (
                    model_type, dataset, self_constructing_var,
                    ("rlr%d" % self_constr_rlr) if should_change_lr
                    else "NOrlr", datetime.now().strftime("%Y_%m_%d_%H%M%S"))
            else:
                self.experiment_id = '%s_%s_k=%d_%s_%s_%s' % (
                    model_type, dataset, growth_rate, layer_num_list,
                    "rlr" if should_change_lr else "NOrlr",
                    datetime.now().strftime("%Y_%m_%d_%H%M%S"))
            # If using a source model, specify that in the ft-logs.
            if source_experiment_id is not None and self.should_save_ft_logs:
                with open(os.path.join(
                        self.save, self.experiment_id + '_ft_log.csv'),
                        'a') as f:
                    f.write(('Using source model \"{}\"\n\n').format(
                             source_experiment_id))

        # Number of classes in the dataset.
        self.num_classes = supported_datasets[dataset_name]
        self.small_inputs = True  # Currently all supported datasets are 32x32

        # Get the right normalize transform for each dataset.
        if dataset_name in ["C10", "C100"]:
            normalize_tfm = transforms.Normalize(*cifar_stats)
        elif dataset_name == "SVHN":
            normalize_tfm = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        # elif dataset_name == "MNIST"
        #     normalize_tfm = transforms.Normalize((*mnist_stats)
        else:
            # Imagenet stats should be good for most problems and datasets.
            normalize_tfm = transforms.Normalize(*imagenet_stats)

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
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_tfm,
            ])

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
            self.train_set = ConcatDataset([
                datasets.SVHN(
                    self.data, split='train', transform=train_transforms,
                    download=not os.path.isdir(self.data)),
                datasets.SVHN(
                    self.data, split='extra', transform=train_transforms,
                    download=False)])
            self.test_set = datasets.SVHN(
                self.data, split='test', transform=test_transforms,
                download=False)

        # If required, define validation set (cut off training set).
        if valid_size:
            indices = torch.randperm(len(self.train_set))
            train_indices = indices[:len(indices) - valid_size]
            valid_indices = indices[len(indices) - valid_size:]
            self.valid_set = Subset(self.train_set, valid_indices)
            self.train_set = Subset(self.train_set, train_indices)
        else:
            self.valid_set = None

        # Define the model to train.
        self.model = DenseNet(
            growth_rate=self.growth_rate,
            block_config=self.block_config,
            bc_mode=self.bc_mode,
            reduction=self.reduction,
            num_init_features=self.growth_rate*2,
            drop_rate=float(1-self.keep_prob),
            num_classes=self.num_classes,
            small_inputs=self.small_inputs,
            efficient=self.efficient,
            seed=self.seed
        )
        print(self.model)
        # Load model state from existing source file if specified.
        if source_experiment_id is not None:
            self.model.load_state_dict(torch.load(os.path.join(
                self.save, source_experiment_id + '_model.pth')))

        # Calculate and print the total number of parameters in the model.
        # TODO: Make a function to count params when DensEMANN is functional.
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
                self.valid_set, batch_size=self.batch_size, shuffle=False,
                pin_memory=(torch.cuda.is_available()), num_workers=0)

        # Move all model parameters and buffers to GPU (if GPU is available).
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # Wrap model for multi-GPUs, if available and necessary.
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Using CUDA with device count = {}".format(
                torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model).cuda()

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
        # If self constructing, create the self-constructing callback
        if self.should_self_construct:
            cbs.append(DensEMANNCallback(
                self_constructing_var=self.self_constructing_var,
                should_change_lr=self.should_change_lr,
                should_save_model=self.should_save_model,
                should_save_ft_logs=self.should_save_ft_logs,
                **self.self_constr_kwargs))
        # If scheduling LR changes, create the callback to do that.
        if self.should_change_lr:
            # The LR schedule length depends on whether or not DensEMANN is
            # used, and on the DensEMANN variant that is used.
            schedule_length = self.n_epochs
            if self.has_micro_algo:
                schedule_length = self.self_constr_kwargs["m_patience_param"]
            elif self.should_self_construct and (
                    self.self_constructing_var >= 2):
                schedule_length = self.self_constr_kwargs["patience_param"]
            cbs.append(ReduceLRCallback(
                rlr_var=self.self_constr_rlr,
                lr=self.lr, gamma=self.gamma,
                rlr_1=self.rlr_1, rlr_2=self.rlr_2,
                schedule_length=schedule_length,
                self_construct_mode=self.should_self_construct))
        # If saving the model, create the callback that saves the best yet.
        if self.should_save_model:
            cbs.append(SaveModelCallback(fname=os.path.join(
                self.save, self.experiment_id + '_model'),
                every_epoch=not valid_loader))
        # If saving ft-logs, create the callback to write the ft-log file.
        if self.should_save_ft_logs:
            cbs.append(CSVLoggerCustom(fname=os.path.join(
                self.save, self.experiment_id + '_ft_log.csv'),
                add_ft_kCS=self.add_ft_kCS,
                ft_comma=self.ft_comma, ft_decimal=self.ft_decimal))

        # Create the Learner from the model, optimizer and callbacks.
        learn = Learner(dls, self.model, lr=self.lr,
                        loss_func=nn.CrossEntropyLoss(),
                        opt_func=opt_func, metrics=[accuracy], cbs=cbs)

        # TRAINING THE MODEL --------------------------------------------------
        # ---------------------------------------------------------------------

        if self.should_train:
            # Initialize best error (useful with validation set).
            best_accuracy = 0
            # Begin counting total training time.
            total_start_time = time.time()
            # Training loop.
            learn.fit(self.lim_n_epochs if self.should_self_construct and (
                self.self_constructing_var >= 2) else self.n_epochs)
            # Measure total training time.
            total_training_time = time.time() - total_start_time
            print("\nTRAINING COMPLETE!\n")
            print("TOTAL TRAINING TIME: %s\n" % str(timedelta(
                seconds=total_training_time)))
            # If DensEMANN was used, print the final architecture.
            if self.should_self_construct:
                total_par, cv_par, fc_par = self.model.count_trainable_params()
                print("Total trainable params: %.1fk" % (total_par / 1e3))
                print("\tConvolutional: %.1fk" % (cv_par / 1e3))
                print("\tFully Connected: %.1fk" % (fc_par / 1e3))
                print("FINAL ARCHITECTURE:\n{}\n".format(self.model))

        # TESTING THE MODEL ---------------------------------------------------
        # ---------------------------------------------------------------------

        if self.should_test:
            # Begin counting testing time.
            test_start_time = time.time()
            # Test on the test set and get results.
            preds, y, losses = learn.get_preds(dl=test_loader, with_loss=True)
            test_time = time.time() - test_start_time
            test_accuracy = float(accuracy(preds, y))
            test_loss = np.mean(losses.tolist())
            print("Test:\taccuracy = %f\tloss = %f" % (
                test_accuracy, test_loss))
            # If required, save relevant data to ft-logs.
            if self.should_save_ft_logs:
                with open(os.path.join(
                        self.save, self.experiment_id + '_ft_log.csv'),
                        'a') as f:
                    to_write = (
                        'test'+'{0}'*2+'{1:0.5f}{0}{2:0.5f}{0}').format(
                        self.ft_comma, test_loss, test_accuracy)
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
                write_at_end += 'training time{0}{1}\n'.format(
                    self.ft_comma, str(timedelta(seconds=total_training_time)))
            if self.should_test:
                write_at_end += 'test time{0}{1}\n\n'.format(
                    self.ft_comma, str(timedelta(seconds=test_time)))
            # If DensEMANN was used, print the final architecture.
            if self.should_self_construct:
                write_at_end += (
                    'total trainable parameters{0}{1}\n' +
                    'convolutional{0}{2}\nfully connected{0}{3}\n').format(
                        self.ft_comma, *self.model.count_trainable_params())
                write_at_end += '\n\"FINAL ARCHITECTURE:\"\n\"{}\"\n'.format(
                    str(self.model).replace('\n', '\"\n\"'))
                with open(os.path.join(
                        self.save, self.experiment_id + '_ft_log.csv'),
                        'a') as f:
                    f.write(write_at_end.replace('.', self.ft_decimal))
