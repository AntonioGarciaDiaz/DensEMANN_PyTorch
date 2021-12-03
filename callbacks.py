# Callbacks for DensEMANN controller,
# for PyTorch + fastai DensEMANN implementation.
# Based on:
# - TensorFlow DensEMANN implementation by me:
#   https://github.com/AntonioGarciaDiaz/Self-constructing_DenseNet
# - PyTorch efficient DenseNet-BC implementation by Geoff Pleiss:
#   https://github.com/gpleiss/efficient_densenet_pytorch
# - TensorFlow DenseNet implementation by Illarion Khlestov:
#   https://github.com/ikhlestov/vision_networks

import numpy as np
import os
import time
import torch
import sys
from collections import deque
from datetime import timedelta
from fastai.vision.all import *
from models import DenseNet


class DensEMANNCallback(Callback):
    """
    Callback for executing the DensEMANN self-construction algorithm.

    Args:
        self_constructing_var (int) - the variant of the self-construction
            algorithm to be used (default is the latest variant).
        should_change_lr (bool) - whether or not the learning rate will be
            modified during training (default True).
        should_save_model (bool) - whether or not to save the model
            (default True).
        should_save_ft_logs (bool) - whether or not to save feature logs
            (default True).
        **kwargs (dict) - contains various optional arguments, all from the
            DensEMANN_controller initializer, that are copied as attributes:
            block_count, layer_cs, asc_thresh, patience_param,
            std_tolerance, std_window, impr_thresh, preserve_transition,
            expansion_rate, dkCS_smoothing, dkCS_std_window, dkCS_stl_thresh,
            auto_usefulness_thresh, auto_uselessness_thresh, m_asc_thresh,
            m_patience_param, complementarity, dont_prune_beyond, acc_lookback,
            and m_re_patience_param.

    Attributes:
        active (bool) - whether or not DensEMANN is currently active or not
            (i.e., whether or not DensEMANN steps are indeed performed after
            each epoch).
        self_constructing_step (function) - function representing a step of the
            chosen DensEMANN variant, executed after each training epoch.
        has_micro_algo (bool) - whether or not the DensEMANN variant in use
            contains a micro-algorithm (i.e. self-constructs at filter level).
        should_change_lr (bool) - from args.
        should_save_model (bool) - from args.
        should_save_ft_logs (bool) - from args.
        expected_end (int) - training epoch at which the algorithm is expected
            to end (used for estimating completion time).
        algorithm_stage (int) - current stage of the algorithm (layer level).
        settled_layers_ceil (int) - highest number of settled layers reached
            as of the current epoch.
        asc_ref_epoch (int) - reference epoch for the ascension stage loop,
            corresponds to the epoch in which the ascension stage began.
        patience_cntdwn (int) - countdown value used for the improvement stage
            (at layer level).
        micro_stage (int) - current stage of the micro-algorithm (filter level)
            if the algorithm has got one.
        useful_filters_ceil (int) - highest number of useful filters reached
            as of the current epoch.
        m_patience_cntdwn (int) - countdown value used for the micro-
            improvement stage (at filter level).
        m_re_patience_cntdwn (int) - countdown value used for the micro-
            recovery stage (at filter level), to terminate it if pre-pruning
            accuracy cannot be reached.
        kCS_list_ref_cntdwn (int) - countdown value used for the micro-
            recovery stage (at filter level); after the countdown,
            the kCS values of filters in the last layer are fixed as reference
            kCS values for these filters until after the next pruning stage.
        accuracy_pre_pruning (float) - accuracy value saved just before the
            last pruning operation took place.
        accuracy_last_layer (float) - accuracy value saved just before the last
            layer was added to the DenseNet.
        accuracy_FIFO (collections.deque) - a double-ended queue containing the
            last accuracies measured on the validation set.
        reduce_lr_callback (ReduceLRCallback or None) - optional reference
            to the ReduceLRCallback if it exists and is being used.
        save_model_callback (SaveModelCallback or None) - optional reference
            to the SaveModelCallback if it exists and is being used.
        csv_logger_callback (CSVLoggerCustom or None) - optional reference
            to the CSVLoggerCustom if it exists and is being used.
        init_num_filters (int) - initial number of convolution filters with
            which the last DenseNet layer was created.
        kCS_list_ref (list of float) - a list of reference kCS values used for
            declaring filters settled, useful, or useless (correspond to the
            current kCS values of filters in the last layer except in long
            micro-recovery stages, to avoid pruning too many filters if their
            kCS values decrease too much).
        kCS_FIFO (list of collections.deque) - a list of double-ended queues
            documenting, for the last few epochs, the evolution of the kCS for
            each filter in the last layer (sampled from kCS_list_ref).
        dkCS_FIFO (list of collections.deque) - a list of double-ended queues
            documenting, for the last few epochs, the evolution of the derivate
            of the kCS for each filter in the last layer.
        usefulness_thresh (float) - usefulness threshold for filters,
            calculated using auto_usefulness_thresh.
        uselessness_thresh (float) - uselessness threshold for filters,
            calculated using auto_uselessness_thresh.
        num_pruned_filters (int) - number of filters removed in the last
            pruning operation.
        Everything in kwargs.
    """
    order = 70

    def __init__(self, self_constructing_var=-1, should_change_lr=True,
                 should_save_model=True, should_save_ft_logs=True, **kwargs):
        """
        Initializer for the DensEMANNCallback.
        """
        self.active = True
        # Use self_constructing_var to select which DensEMANN variant to use.
        if self_constructing_var >= 0 and self_constructing_var <= 7:
            exec(("self.self_constructing_step = self.self_constructing_var{}"
                  ).format(self_constructing_var))
        elif self_constructing_var == 1000:
            self.self_constructing_step = self.self_constructing_minimal
        # Deduce if the DensEMANN variant has got a micro-algorithm.
        self.has_micro_algo = self_constructing_var >= 4

        # Copy most of the attributes from args (and kwargs).
        self.should_change_lr = should_change_lr
        self.should_save_model = should_save_model
        self.should_save_ft_logs = should_save_ft_logs
        self.__dict__.update(kwargs)
        # Initialise the algorithm variables.
        self.initialise_algorithm_variables()
        # These arguments will be initialised just before fit
        self.reduce_lr_callback = None
        self.save_model_callback = None
        self.csv_logger_callback = None

    def set_expected_end(self, expected_end):
        """
        Set the expected last training epoch for the algorithm.
        On basis of this, estimate the time left until the algorithm's
        completion, and print it on console.

        Args:
            expected_end (int) - new training epoch at which the algorithm is
                expected to end.
        """
        self.expected_end = expected_end
        # Get the execution time for the last training epoch.
        time_per_epoch = time.time() - self.recorder.start_epoch
        # Use it to deduce the time left until completion (in seconds).
        seconds_left = int(
            (self.expected_end - self.learn.epoch) * time_per_epoch)
        print("Completion (epoch %d) expected in: %s." % (
            self.expected_end, str(timedelta(seconds=seconds_left))))

    def set_algorithm_stage(self, algorithm_stage=None, micro_stage=None):
        """
        Set the algorithm and micro-algorithm's stage, and print the current
        stage on console. None values mean that the current stage continues.

        Args:
            algorithm_stage (int or None) - identifier for the new algorithm
                stage, at layer level (default None, i.e. the current stage
                continues).
            micro_stage (int or None) - identifier for the new micro-algorithm
                stage, at filter level (default None, i.e. the current stage
                continues).
        """
        # Set algorithm stage.
        if algorithm_stage is not None:
            self.algorithm_stage = algorithm_stage
            if self.algorithm_stage == 0:
                print("-------------- ASCENSION STAGE --------------")
            elif self.algorithm_stage == 1:
                print("------------- IMPROVEMENT STAGE -------------")
            elif self.algorithm_stage == 2:
                print("---------------- FINAL STAGE ----------------")
        # Set micro-algorithm stage.
        if micro_stage is not None:
            self.micro_stage = micro_stage
            # Micro-stage labels are only printed during improvement stage.
            if self.algorithm_stage == 1:
                if self.micro_stage == 0:
                    print("----------- Micro-Ascension Stage -----------")
                elif self.micro_stage == 1:
                    print("---------- Micro-Improvement Stage ----------")
                elif self.micro_stage == 2:
                    print("------------ Micro-Pruning Stage ------------")
                elif self.micro_stage == 3:
                    print("----------- Micro-Recovery Stage ------------")
                elif self.micro_stage == 4:
                    print("------------- Micro-Final Stage -------------")

    def initialise_algorithm_variables(self, asc_ref_epoch=0):
        """
        Initialise or reset the measurable variables used by the algorithm,
        with their initial values at the start of the algorithm's execution.

        Args:
            asc_ref_epoch (int) - reference epoch for the ascension stage loop,
                should correspond to the epoch in which the ascension stage
                began, but may be set manually here (default 0).
        """
        self.expected_end = self.patience_param
        self.set_algorithm_stage(algorithm_stage=0)
        self.settled_layers_ceil = 0  # highest num of settled layers yet
        self.asc_ref_epoch = asc_ref_epoch
        self.patience_cntdwn = self.patience_param
        if self.has_micro_algo:
            self.expected_end = self.m_patience_param
            self.set_algorithm_stage(micro_stage=0)
            self.useful_filters_ceil = 0  # highest num of useful filters yet
            self.m_patience_cntdwn = self.m_patience_param
            self.m_re_patience_cntdwn = self.m_re_patience_param
            self.kCS_list_ref_cntdwn = self.m_patience_param
            self.accuracy_pre_pruning = 0
            self.accuracy_last_layer = 0
            self.accuracy_FIFO = deque(
                maxlen=max(self.std_window, self.acc_lookback))
        else:
            self.accuracy_FIFO = deque(maxlen=self.std_window)

    def before_fit(self):
        """
        Before the training process begins, a few more initialisation tasks
        must take place.
        """
        # Gather references to optional callbacks, which can be useful later.
        if self.should_change_lr:
            self.reduce_lr_callback = next(
                c for c in self.learn.cbs if isinstance(c, ReduceLRCallback))
        if self.should_save_model:
            self.save_model_callback = next(
                c for c in self.learn.cbs if isinstance(c, SaveModelCallback))
        if self.should_save_ft_logs:
            self.csv_logger_callback = next(
                c for c in self.learn.cbs if isinstance(c, CSVLoggerCustom))
        # Keep references to some of the DenseNet model attributes.
        self.init_num_filters = self.learn.model.growth_rate
        # kCS and dkCS FIFO lists are only used in variants with a
        # micro-algorithm (i.e. with self-construction at filter level).
        if self.has_micro_algo:
            self.kCS_FIFO = [deque(maxlen=self.dkCS_smoothing)
                             for i in range(self.init_num_filters)]
            self.dkCS_FIFO = [deque(maxlen=self.dkCS_std_window)
                              for i in range(self.init_num_filters)]

    def post_self_construction_routine(self):
        """
        Routine to follow after every self-construction operation.
        Move the new model to GPU, then save it using the SaveModelCallback.
        Also rewrite the optimizer's parameter groups to reflect the new model.
        """
        # Move model to GPU (if GPU is available).
        if torch.cuda.is_available():
            self.learn.model = self.learn.model.cuda()
        # Wrap model for multi-GPUs, if available and necessary.
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.learn.model = torch.nn.DataParallel(
                self.learn.model).cuda()
        # Save the model using the SaveModelCallback.
        if self.should_save_model:
            self.save_model_callback._save(f'{self.save_model_callback.fname}')

        # Display the new total number of parameters.
        num_params = sum(p.numel() for p in self.model.parameters())
        print("Total parameters: ", num_params)

        # Re-initialize the optimizer's parameter groups.
        # First, get the default hypers (from the first parameter group).
        def_hypers = {k: v for k, v in self.opt.opt.param_groups[0].items() if
                      k != 'params'}
        # Then turn each parameter in the model into a parameter group
        # (consisting of the parameter + the optimizer's default hypers).
        param_groups = []
        for param in list(self.learn.model.parameters()):
            pg = {'params': [param]}
            pg.update(def_hypers)
            param_groups.append(pg)
        # Finally, replace the old param_groups with the new parameter groups.
        self.opt.opt.param_groups = param_groups

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
        # Run the DenseNet model's own add_new_filters method.
        self.learn.model.add_new_filters(
            num_new_filters=num_new_filters, complementarity=complementarity,
            preserve_transition=preserve_transition)
        # Execute the post-self-construction routine.
        self.post_self_construction_routine()

        # Modify the FIFO lists to reflect the new filter's addition.
        self.kCS_FIFO.extend([deque(maxlen=self.dkCS_smoothing)
                              for i in range(num_new_filters)])
        self.dkCS_FIFO.extend([deque(maxlen=self.dkCS_std_window)
                               for i in range(num_new_filters)])

        # Announce the new filter's addition by printing it on console.
        print("ADDED A NEW FILTER TO LAYER #%d (BLOCK #%d)!" %
              (self.learn.model.block_config[-1],
               len(self.learn.model.block_config)))

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
        # If saving the model, save its pre-pruning state separately.
        if self.should_save_model:
            self.num_pruned_filters = len(filter_ids)
            self.learn.save(f'{self.save_model_callback.fname}_prepruning',
                            with_opt=self.save_model_callback.with_opt)

        # Run the DenseNet model's own remove_filters method.
        self.learn.model.remove_filters(
            filter_ids, preserve_transition=preserve_transition)
        # Execute the post-self-construction routine.
        self.post_self_construction_routine()

        # Modify the FIFO lists to reflect the filters' removal.
        for i in reversed(filter_ids):
            del self.kCS_FIFO[i], self.dkCS_FIFO[i]

        # Announce the filters' removal by printing it on console.
        print("PRUNED FILTERS (%s) IN LAYER #%d (BLOCK #%d)!" %
              (', '.join(map(str, filter_ids)),
               self.learn.model.block_config[-1],
               len(self.learn.model.block_config)))
        # Write which filters were pruned in the feature logs.
        if self.should_save_ft_logs:
            self.csv_logger_callback.file.write(
                '\"Pruned: {}\"\n'.format(filter_ids))

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
        # Run the DenseNet model's own add_new_layers method.
        self.learn.model.add_new_layers(
            num_new_layers=num_new_layers, growth_rate=growth_rate,
            preserve_transition=preserve_transition, efficient=efficient)
        # Execute the post-self-construction routine.
        self.post_self_construction_routine()

        # Interpret None values for the growth rate.
        if growth_rate is None:
            growth_rate = self.learn.model.growth_rate
        # Copy the growth rate value as the new initial number of filters.
        self.init_num_filters = growth_rate

        # If self-constructing at filter level, reset the FIFO lists.
        if self.has_micro_algo:
            self.kCS_FIFO = [deque(maxlen=self.dkCS_smoothing)
                             for i in range(self.init_num_filters)]
            self.dkCS_FIFO = [deque(maxlen=self.dkCS_std_window)
                              for i in range(self.init_num_filters)]

        # Announce the new layer's addition by printing it on console.
        if self.learn.model.bc_mode:
            print("ADDED %d NEW BOTTLENECK AND %d NEW COMPOSITE LAYERS "
                  "(%d filters) to the last block (#%d)! "
                  "It now has got %d bottleneck and %d composite layers." %
                  (num_new_layers, num_new_layers, growth_rate,
                   len(self.learn.model.block_config),
                   self.learn.model.block_config[-1],
                   self.learn.model.block_config[-1]))
        else:
            print("ADDED %d NEW LAYERS (%d filters) to the last block (#%d)! "
                  "It now has got %d layers." %
                  (num_new_layers, growth_rate,
                   len(self.learn.model.block_config),
                   self.learn.model.block_config[-1]))

    def add_new_block(self, num_layers=1, growth_rate=None, efficient=None):
        """
        Add a new dense block in the DenseNet, with a given number of layers
        and growth rate.

        Args:
            num_layers (int) - number of layers in the new block (default 1).
            growth_rate (int or None) - number of filters in the new layers,
                (default None, i.e. the DenseNet's growth_rate attribute).
            efficient (bool) - set to True to use checkpointing
                (default None, i.e. use the value provided at creation).
        """
        # Run the DenseNet model's own add_new_block method.
        self.learn.model.add_new_block(
            num_layers=num_layers, growth_rate=growth_rate,
            efficient=efficient)
        # Execute the post-self-construction routine.
        self.post_self_construction_routine()

        # Interpret None values for the growth rate.
        if growth_rate is None:
            growth_rate = self.learn.model.growth_rate
        # Copy the growth rate value as the new initial number of filters.
        self.init_num_filters = growth_rate

        # If self-constructing at filter level, reset the FIFO lists.
        if self.has_micro_algo:
            self.kCS_FIFO = [deque(maxlen=self.dkCS_smoothing)
                             for i in range(self.init_num_filters)]
            self.dkCS_FIFO = [deque(maxlen=self.dkCS_std_window)
                              for i in range(self.init_num_filters)]
        print("ADDED A NEW BLOCK (#%d)!" % len(self.learn.model.block_config))
        print("The current architecture is:\n{}".format(self.model))

    def self_constructing_var4(self, epoch, accuracy):
        """
        A step of DensEMANN variant #4 for one training epoch.
        Builds new layers in the last block depending on parameters.
        Returns True if training should continue, False otherwise.

        This algorithm consists of an macro-algorithm, which adds layers to the
        last block, and a micro-algorithm, which builds those layers filter by
        filter.
        - The macro-algorithm calls the micro-algorithm to build the last layer
          in the block (i.e. to modify it by adding/pruning filters in it).
          It then checks if the accuracy has improved significantly since the
          previous layer's addition, or since the beginning of the training if
          no new layers have been added yet.
          If so it adds a new layer (with the specified growth rate) and starts
          over, else the algorithm ends.
        - The micro-algorithm consists of a succession of four stages:
            - Ascension: add one filter every m_asc_thresh training epochs,
              break the loop (end the stage) when one of the filters settles
              (its CS has remained stable for a certain number of epochs).
            - Improvement: countdown of m_patience_param epochs; if the number
              of useful filters (CS above usefulness_thresh, automatically set)
              is above the latest max number of useful filters, add a filter
              and restart the countdown; if the countdown ends, wait until all
              filters have settled and end the stage.
            - Pruning: save the current accuracy and prune all useless filters
              (CS below uselessness_thresh, automatically set).
            - Recovery: wait for one last countdown of m_patience_param epochs
              (optionally resetting the learning rate to its initial value and
              reducing it according to rlr0); after this countdown wait until
              reaching pre-pruning accuracy, then end the stage.

        Args:
            epoch (int) - current training epoch (since adding the last block);
            accuracy (float) -  accuracy for this epoch.

        Returns:
            continue_training (bool) - whether or not the algorithm should end
                after this step.
        """
        continue_training = True

        settled_filters_count = 0
        useful_filters_count = 0
        useless_filters_list = []
        kCS_settled = []
        # Update the filter kCS lists, count settled and useful filters
        kCS_list = self.learn.model.get_kCS_list_from_layer(-1, -1)
        # The actual kCS list for counting settled, useful and useless filters
        # is only updated if the associated countdown has not ended
        if self.kCS_list_ref_cntdwn > 0:
            self.kCS_list_ref = kCS_list
        for k in range(len(self.kCS_FIFO)):
            self.kCS_FIFO[k].append(self.kCS_list_ref[k])
            if len(self.kCS_FIFO[k]) == self.dkCS_smoothing:
                self.dkCS_FIFO[k].append(
                    (self.kCS_FIFO[k][-1] - self.kCS_FIFO[k][0])/(
                        self.dkCS_smoothing-1))
                # Settled = dkCS remained close to 0 during the last epochs
                if ((len(self.dkCS_FIFO[k]) == self.dkCS_std_window) and (
                        np.abs(np.mean(self.dkCS_FIFO[k])
                               ) <= self.dkCS_stl_thresh) and
                        (np.abs(np.std(self.dkCS_FIFO[k])
                                ) <= self.dkCS_stl_thresh)):
                    settled_filters_count += 1
                    if self.micro_stage == 1:
                        kCS_settled.append(self.kCS_FIFO[k][-1])

        # If half of the original filters have settled
        if settled_filters_count >= 0.5*self.init_num_filters:
            # During impr. stage, calculate UFT and ULT
            if self.micro_stage == 1:
                self.usefulness_thresh = min(kCS_settled) + (
                    max(kCS_settled) - min(kCS_settled)
                    )*self.auto_usefulness_thresh
                self.uselessness_thresh = min(kCS_settled) + (
                    max(kCS_settled) - min(kCS_settled)
                    )*self.auto_uselessness_thresh
            # Detect and count useful and useless filters
            for k in range(len(self.kCS_FIFO)):
                # Useful = kCS above the usefulness thresh
                if np.mean(self.kCS_FIFO[k]) >= self.usefulness_thresh:
                    useful_filters_count += 1
                # Useless = kCS below the uselessness thresh
                if np.mean(self.kCS_FIFO[k]) <= self.uselessness_thresh:
                    useless_filters_list.append(k)

        # stage #0 = ascension stage (currently does nothing)
        if self.algorithm_stage == 0:
            print("(This variant hasn't got an ascension stage,"
                  " skipping to improvement.)")
            self.set_algorithm_stage(algorithm_stage=1, micro_stage=0)

        # stage #1 = improvement stage
        if self.algorithm_stage == 1:
            # micro-stage #0 = micro-ascension stage
            if self.micro_stage == 0:
                if settled_filters_count >= 1:
                    # end stage when one or various filters have settled
                    self.useful_filters_ceil = useful_filters_count
                    self.set_algorithm_stage(micro_stage=1)
                    if self.should_change_lr:
                        self.reduce_lr_callback.activation_switch(
                            reset_lr=True)
                    self.set_expected_end(epoch + 2*(self.m_patience_param+1))
                elif (epoch-self.asc_ref_epoch+1) % self.m_asc_thresh == 0:
                    self.add_new_filters(
                        num_new_filters=self.expansion_rate,
                        complementarity=self.complementarity,
                        preserve_transition=self.preserve_transition)

            # micro-stage #1 = micro-improvement stage
            if self.micro_stage == 1:
                if self.m_patience_cntdwn <= 0 and (
                        settled_filters_count == len(self.kCS_FIFO)):
                    # at the end of the patience countdown, end stage when all
                    # the filters have settled
                    self.set_algorithm_stage(micro_stage=2)
                elif useful_filters_count > self.useful_filters_ceil:
                    # if the number of useful filters is above the latest max,
                    # add a filter and restart ctdwn
                    self.useful_filters_ceil = useful_filters_count
                    self.add_new_filters(
                        num_new_filters=self.expansion_rate,
                        complementarity=self.complementarity,
                        preserve_transition=self.preserve_transition)
                    self.m_patience_cntdwn = self.m_patience_param
                    if self.should_change_lr:
                        self.reduce_lr_callback.activation_switch()
                    self.set_expected_end(epoch + 2*(self.m_patience_param+1))
                else:
                    # patience countdown progress
                    self.m_patience_cntdwn -= 1

            # micro-stage #2 = micro-pruning stage
            if self.micro_stage == 2:
                # if no filters must be pruned, or if less than
                # dont_prune_beyond filters would remain after the pruning
                # operation, skip directly to the final stage
                if len(useless_filters_list) == 0 or (
                        len(useless_filters_list) >
                        len(self.kCS_FIFO) - self.dont_prune_beyond):
                    if self.should_change_lr:
                        self.reduce_lr_callback.deactivation_switch()
                    if self.should_save_model:
                        self.save_model_callback._save(
                            f'{self.save_model_callback.fname}')
                    self.set_algorithm_stage(micro_stage=4)
                # else, save the accuracy, prune useless filters and end stage
                self.accuracy_pre_pruning = max([
                    self.accuracy_FIFO[-i-1] for i in range(min(
                        self.acc_lookback, len(self.accuracy_FIFO)))])
                self.remove_filters(
                    filter_ids=useless_filters_list,
                    preserve_transition=self.preserve_transition)
                self.set_algorithm_stage(micro_stage=3)
                # run one last patience countdown for recovery
                self.m_patience_cntdwn = self.m_patience_param
                if self.should_change_lr:
                    self.reduce_lr_callback.activation_switch(reset_lr=True)
                # activate the countdown for keeping the kCS list as reference
                self.kCS_list_ref_cntdwn = (
                    self.m_patience_param - self.m_patience_cntdwn)
                self.set_expected_end(epoch + self.m_patience_param + 1)

            # micro-stage #3 = micro-recovery stage (accessed in next epoch)
            elif self.micro_stage == 3:
                # wait until reaching pre-pruning accuracy, then end the stage
                if self.m_patience_cntdwn <= 0 and (
                        accuracy >= self.accuracy_pre_pruning):
                    if self.should_change_lr:
                        self.reduce_lr_callback.deactivation_switch()
                    if self.should_save_model:
                        self.save_model_callback._save(
                            f'{self.save_model_callback.fname}')
                    self.set_algorithm_stage(micro_stage=4)
                # if the stage lasts too much time, end it
                elif self.m_re_patience_cntdwn <= 0:
                    if self.should_change_lr:
                        self.reduce_lr_callback.deactivation_switch()
                    # undo the previous pruning-recovery if saving the model
                    if self.should_save_model:
                        print("Restoring model state before previous pruning.")
                        self.add_new_filters(
                            num_new_filters=self.num_pruned_filters,
                            complementarity=False,
                            preserve_transition=self.preserve_transition)
                        print("Loading back pre-pruning weights.")
                        self.learn.load(
                            f'{self.save_model_callback.fname}_prepruning',
                            with_opt=self.save_model_callback.with_opt)
                        self.save_model_callback._save(
                            f'{self.save_model_callback.fname}')
                    self.set_algorithm_stage(micro_stage=4)
                # kCS ref and patience countdown progress
                # (at the end the kCS values remain fixed)
                self.kCS_list_ref_cntdwn -= 1
                self.m_patience_cntdwn -= 1
                self.m_re_patience_cntdwn -= 1

            # at the end of the micro-algorithm, try to add a new layer
            if self.micro_stage == 4:
                # reset everything for the micro-algorithm
                self.micro_stage = 0
                self.useful_filters_ceil = 0
                self.m_patience_cntdwn = self.m_patience_param
                self.m_re_patience_cntdwn = self.m_re_patience_param
                self.kCS_list_ref_cntdwn = self.m_patience_param
                self.accuracy_pre_pruning = 0
                # check if the accuracy has improved since the last layer
                # if so, add a layer, else go to the final stage
                if abs(accuracy-self.accuracy_last_layer) >= self.impr_thresh:
                    self.accuracy_last_layer = accuracy
                    self.add_new_layers(
                        preserve_transition=self.preserve_transition)
                    # alt. number of filters = half the previous
                    # layer's number if during the ascension stage.
                    #     growth_rate=floor(
                    #         len(self.filters_ref_list[-1][-1])/2))
                    self.set_algorithm_stage(micro_stage=0)
                    self.set_expected_end(epoch + 2*(self.m_patience_param+1))
                else:
                    self.set_algorithm_stage(algorithm_stage=2)

        # stage #2 = end (stop the algorithm and reset everything)
        if self.algorithm_stage == 2:
            continue_training = False
            self.algorithm_stage = 0
            self.patience_cntdwn = self.patience_param
            self.accuracy_last_layer = 0

        return continue_training

    def self_constructing_var5(self, epoch, accuracy):
        """
        A step of DensEMANN variant #5 for one training epoch.
        Builds new layers in the last block depending on parameters.
        Returns True if training should continue, False otherwise.

        This algorithm consists of an macro-algorithm, which adds layers to the
        last block, and a micro-algorithm, which builds those layers filter by
        filter.
        - The macro-algorithm calls the micro-algorithm to build the last layer
          in the block (i.e. to modify it by adding/pruning filters in it).
          It then checks if the accuracy has improved significantly since the
          previous layer's addition, or since the beginning of the training if
          no new layers have been added yet.
          If so it adds a new layer (with the specified growth rate) and starts
          over, else the algorithm ends.
        - The micro-algorithm consists of a succession of three stages:
            - Improvement: countdown of m_patience_param epochs; if the number
              of useful filters (CS above usefulness_thresh, automatically set)
              is above the latest max number of useful filters, add a filter
              and restart the countdown; if the countdown ends, wait until all
              filters have settled and end the stage.
            - Pruning: save the current accuracy and prune all useless filters
              (CS below uselessness_thresh, automatically set).
            - Recovery: wait for one last countdown of m_patience_param epochs
              (optionally resetting the learning rate to its initial value and
              reducing it according to rlr0); after this countdown wait until
              reaching pre-pruning accuracy, then if there are any new useless
              filters wait for all filters to settle and return to pruning,
              else end the stage.

        Args:
            epoch (int) - current training epoch (since adding the last block);
            accuracy (float) -  accuracy for this epoch.

        Returns:
            continue_training (bool) - whether or not the algorithm should end
                after this step.
        """
        continue_training = True

        settled_filters_count = 0
        useful_filters_count = 0
        useless_filters_list = []
        kCS_settled = []
        # Update the filter kCS lists, count settled and useful filters
        kCS_list = self.learn.model.get_kCS_list_from_layer(-1, -1)
        # The actual kCS list for counting settled, useful and useless filters
        # is only updated if the associated countdown has not ended
        if self.kCS_list_ref_cntdwn > 0:
            self.kCS_list_ref = kCS_list
        for k in range(len(self.kCS_FIFO)):
            self.kCS_FIFO[k].append(self.kCS_list_ref[k])
            if len(self.kCS_FIFO[k]) == self.dkCS_smoothing:
                self.dkCS_FIFO[k].append(
                    (self.kCS_FIFO[k][-1] - self.kCS_FIFO[k][0])/(
                        self.dkCS_smoothing-1))
                # Settled = dkCS remained close to 0 during the last epochs
                if ((len(self.dkCS_FIFO[k]) == self.dkCS_std_window) and (
                        np.abs(np.mean(self.dkCS_FIFO[k])
                               ) <= self.dkCS_stl_thresh) and
                        (np.abs(np.std(self.dkCS_FIFO[k])
                                ) <= self.dkCS_stl_thresh)):
                    settled_filters_count += 1
                    if self.micro_stage == 1:
                        kCS_settled.append(self.kCS_FIFO[k][-1])

        # If half of the original filters have settled
        if settled_filters_count >= 0.5*self.init_num_filters:
            # During impr. stage, calculate UFT and ULT
            if self.micro_stage == 1:
                self.usefulness_thresh = min(kCS_settled) + (
                    max(kCS_settled) - min(kCS_settled)
                    )*self.auto_usefulness_thresh
                self.uselessness_thresh = min(kCS_settled) + (
                    max(kCS_settled) - min(kCS_settled)
                    )*self.auto_uselessness_thresh
            # Detect and count useful and useless filters
            for k in range(len(self.kCS_FIFO)):
                # Useful = kCS above the usefulness thresh
                if np.mean(self.kCS_FIFO[k]) >= self.usefulness_thresh:
                    useful_filters_count += 1
                # Useless = kCS below the uselessness thresh
                if np.mean(self.kCS_FIFO[k]) <= self.uselessness_thresh:
                    useless_filters_list.append(k)

        # stage #0 = ascension stage (currently does nothing)
        if self.algorithm_stage == 0:
            print("(This variant hasn't got an ascension stage,"
                  " skipping to improvement.)")
            self.set_algorithm_stage(algorithm_stage=1, micro_stage=1)
            if self.should_change_lr:
                self.reduce_lr_callback.activation_switch(reset_lr=True)
            self.set_expected_end(epoch + 2*(self.m_patience_param+1))

        # stage #1 = improvement stage
        if self.algorithm_stage == 1:
            # micro-stage #1 = micro-improvement stage
            if self.micro_stage == 1:
                if self.m_patience_cntdwn <= 0 and (
                        settled_filters_count == len(self.kCS_FIFO)):
                    # at the end of the patience countdown, end stage when all
                    # the filters have settled
                    self.set_algorithm_stage(micro_stage=2)
                elif useful_filters_count > self.useful_filters_ceil:
                    # if the number of useful filters is above the latest max,
                    # add a filter and restart ctdwn
                    self.useful_filters_ceil = useful_filters_count
                    self.add_new_filters(
                        num_new_filters=self.expansion_rate,
                        complementarity=self.complementarity,
                        preserve_transition=self.preserve_transition)
                    self.m_patience_cntdwn = self.m_patience_param
                    if self.should_change_lr:
                        self.reduce_lr_callback.activation_switch()
                    self.set_expected_end(epoch + 2*(self.m_patience_param+1))
                else:
                    # patience countdown progress
                    self.m_patience_cntdwn -= 1

            # micro-stage #2 = micro-pruning stage
            if self.micro_stage == 2:
                # if no filters must be pruned, or if less than
                # dont_prune_beyond filters would remain after the pruning
                # operation, skip directly to the final stage
                if len(useless_filters_list) == 0 or (
                        len(useless_filters_list) >
                        len(self.kCS_FIFO) - self.dont_prune_beyond):
                    if self.should_change_lr:
                        self.reduce_lr_callback.deactivation_switch()
                    if self.should_save_model:
                        self.save_model_callback._save(
                            f'{self.save_model_callback.fname}')
                    self.set_algorithm_stage(micro_stage=4)
                # else, save the accuracy, prune useless filters and end stage
                self.accuracy_pre_pruning = max([
                    self.accuracy_FIFO[-i-1] for i in range(min(
                        self.acc_lookback, len(self.accuracy_FIFO)))])
                self.remove_filters(
                    filter_ids=useless_filters_list,
                    preserve_transition=self.preserve_transition)
                self.set_algorithm_stage(micro_stage=3)
                # run one last patience countdown for recovery
                self.m_patience_cntdwn = self.m_patience_param
                if self.should_change_lr:
                    self.reduce_lr_callback.activation_switch(reset_lr=True)
                # activate the countdown for keeping the kCS list as reference
                self.kCS_list_ref_cntdwn = (
                    self.m_patience_param - self.m_patience_cntdwn)
                self.set_expected_end(epoch + self.m_patience_param + 1)

            # micro-stage #3 = micro-recovery stage (accessed in next epoch)
            elif self.micro_stage == 3:
                # wait until reaching pre-pruning accuracy, then end the stage
                if self.m_patience_cntdwn <= 0 and (
                        accuracy >= self.accuracy_pre_pruning):
                    # prune again if there are useless filters, else continue
                    # but first, wait for all filters to settle
                    if len(useless_filters_list) >= 1 and (
                            settled_filters_count == len(self.kCS_FIFO)):
                        self.set_algorithm_stage(micro_stage=2)
                    else:
                        if self.should_change_lr:
                            self.reduce_lr_callback.deactivation_switch()
                        if self.should_save_model:
                            self.save_model_callback._save(
                                f'{self.save_model_callback.fname}')
                        self.set_algorithm_stage(micro_stage=4)
                # if the stage lasts too much time, end it
                elif self.m_re_patience_cntdwn <= 0:
                    if self.should_change_lr:
                        self.reduce_lr_callback.deactivation_switch()
                    # undo the previous pruning-recovery if saving the model
                    if self.should_save_model:
                        print("Restoring model state before previous pruning.")
                        self.add_new_filters(
                            num_new_filters=self.num_pruned_filters,
                            complementarity=False,
                            preserve_transition=self.preserve_transition)
                        print("Loading back pre-pruning weights.")
                        self.learn.load(
                            f'{self.save_model_callback.fname}_prepruning',
                            with_opt=self.save_model_callback.with_opt)
                        self.save_model_callback._save(
                            f'{self.save_model_callback.fname}')
                    self.set_algorithm_stage(micro_stage=4)
                # kCS ref and patience countdown progress
                # (at the end the kCS values remain fixed)
                self.kCS_list_ref_cntdwn -= 1
                self.m_patience_cntdwn -= 1
                self.m_re_patience_cntdwn -= 1

            # at the end of the micro-algorithm, try to add a new layer
            if self.micro_stage == 4:
                # reset everything for the micro-algorithm
                self.micro_stage = 0
                self.useful_filters_ceil = 0
                self.m_patience_cntdwn = self.m_patience_param
                self.m_re_patience_cntdwn = self.m_re_patience_param
                self.kCS_list_ref_cntdwn = self.m_patience_param
                self.accuracy_pre_pruning = 0
                # check if the accuracy has improved since the last layer
                # if so, add a layer, else go to the final stage
                if abs(accuracy-self.accuracy_last_layer) >= self.impr_thresh:
                    self.accuracy_last_layer = accuracy
                    self.add_new_layers(
                        preserve_transition=self.preserve_transition)
                    # alt. number of filters = half the previous
                    # layer's number if during the ascension stage.
                    #     growth_rate=floor(
                    #         len(self.filters_ref_list[-1][-1])/2))
                    self.set_algorithm_stage(micro_stage=1)
                    self.set_expected_end(epoch + 2*(self.m_patience_param+1))
                    if self.should_change_lr:
                        self.reduce_lr_callback.activation_switch(
                            reset_lr=True)
                else:
                    self.set_algorithm_stage(algorithm_stage=2)

        # stage #2 = end (stop the algorithm and reset everything)
        if self.algorithm_stage == 2:
            continue_training = False
            self.algorithm_stage = 0
            self.patience_cntdwn = self.patience_param
            self.accuracy_last_layer = 0

        return continue_training

    def self_constructing_var6(self, epoch, accuracy):
        """
        A step of DensEMANN variant #6 for one training epoch.
        Builds new layers in the last block depending on parameters.
        Returns True if training should continue, False otherwise.

        This algorithm consists of an macro-algorithm, which adds layers to the
        last block, and a micro-algorithm, which builds those layers filter by
        filter.
        - The macro-algorithm calls the micro-algorithm to build the last layer
          in the block (i.e. to modify it by adding/pruning filters in it).
          It then checks if the accuracy has improved significantly since the
          previous layer's addition, or since the beginning of the training if
          no new layers have been added yet.
          If so it adds a new layer (with the specified growth rate) and starts
          over, else the algorithm ends.
        - The micro-algorithm consists of a succession of three stages:
            - Improvement: countdown of m_patience_param epochs; if a new
              (settled) filter becomes useful (CS above usefulness_thresh,
              automatically set), add a filter and restart the countdown;
              if the countdown ends, wait until all filters have settled and
              end the stage.
            - Pruning: save the current accuracy and prune all useless filters
              (CS below uselessness_thresh, automatically set).
            - Recovery: if/when the learning reate is at its minimal value
              (optionally after resetting the learning rate to its initial
              value and reducing it according to rlr0), wait until reaching
              pre-pruning accuracy; then if there are any new useless filters
              wait for all filters to settle and return to pruning, else end
              the stage.

        Args:
            epoch (int) - current training epoch (since adding the last block);
            accuracy (float) -  accuracy for this epoch.

        Returns:
            continue_training (bool) - whether or not the algorithm should end
                after this step.
        """
        continue_training = True

        settled_filters_list = []
        useful_filters_count = 0
        useless_filters_list = []
        kCS_settled = []
        # Update the filter kCS lists, count settled and useful filters
        kCS_list = self.learn.model.get_kCS_list_from_layer(-1, -1)
        # The actual kCS list for counting settled, useful and useless filters
        # is only updated if the associated countdown has not ended
        if self.kCS_list_ref_cntdwn > 0:
            self.kCS_list_ref = kCS_list
        for k in range(len(self.kCS_FIFO)):
            self.kCS_FIFO[k].append(self.kCS_list_ref[k])
            if len(self.kCS_FIFO[k]) == self.dkCS_smoothing:
                self.dkCS_FIFO[k].append(
                    (self.kCS_FIFO[k][-1] - self.kCS_FIFO[k][0])/(
                        self.dkCS_smoothing-1))
                # Settled = dkCS remained close to 0 during the last epochs
                if ((len(self.dkCS_FIFO[k]) == self.dkCS_std_window) and (
                        np.abs(np.mean(self.dkCS_FIFO[k])
                               ) <= self.dkCS_stl_thresh) and
                        (np.abs(np.std(self.dkCS_FIFO[k])
                                ) <= self.dkCS_stl_thresh)):
                    settled_filters_list.append(k)
                    if self.micro_stage == 1:
                        kCS_settled.append(self.kCS_FIFO[k][-1])

        # If half of the original filters have settled
        if len(settled_filters_list) >= 0.5*self.init_num_filters:
            # During impr. stage, calculate UFT and ULT
            if self.micro_stage == 1:
                self.usefulness_thresh = min(kCS_settled) + (
                    max(kCS_settled) - min(kCS_settled)
                    )*self.auto_usefulness_thresh
                self.uselessness_thresh = min(kCS_settled) + (
                    max(kCS_settled) - min(kCS_settled)
                    )*self.auto_uselessness_thresh
            # Detect and count useful and useless filters
            for k in settled_filters_list:
                # Useful = settled, and kCS above the usefulness thresh
                if np.mean(self.kCS_FIFO[k]) >= self.usefulness_thresh:
                    useful_filters_count += 1
                # Useless = settled, and kCS below the uselessness thresh
                if np.mean(self.kCS_FIFO[k]) <= self.uselessness_thresh:
                    useless_filters_list.append(k)

        # stage #0 = ascension stage (currently does nothing)
        if self.algorithm_stage == 0:
            print("(This variant hasn't got an ascension stage,"
                  " skipping to improvement.)")
            self.set_algorithm_stage(algorithm_stage=1, micro_stage=1)
            if self.should_change_lr:
                self.reduce_lr_callback.activation_switch(reset_lr=True)
            self.set_expected_end(epoch + 2*(self.m_patience_param+1))

        # stage #1 = improvement stage
        if self.algorithm_stage == 1:
            # micro-stage #1 = micro-improvement stage
            if self.micro_stage == 1:
                if self.m_patience_cntdwn <= 0 and (
                        len(settled_filters_list) == len(self.kCS_FIFO)):
                    # at the end of the patience countdown, end stage when all
                    # the filters have settled
                    self.set_algorithm_stage(micro_stage=2)
                elif useful_filters_count > self.useful_filters_ceil:
                    # if a new filter is useful, add a filter and restart ctdwn
                    self.useful_filters_ceil = useful_filters_count
                    self.add_new_filters(
                        num_new_filters=self.expansion_rate,
                        complementarity=self.complementarity,
                        preserve_transition=self.preserve_transition)
                    self.m_patience_cntdwn = self.m_patience_param
                    if self.should_change_lr:
                        self.reduce_lr_callback.activation_switch()
                    self.set_expected_end(epoch + 2*(self.m_patience_param+1))
                else:
                    # patience countdown progress
                    self.m_patience_cntdwn -= 1

            # micro-stage #2 = micro-pruning stage
            if self.micro_stage == 2:
                # if no filters must be pruned, or if less than
                # dont_prune_beyond filters would remain after the pruning
                # operation, skip directly to the final stage
                if len(useless_filters_list) == 0 or (
                        len(useless_filters_list) >
                        len(self.kCS_FIFO) - self.dont_prune_beyond):
                    if self.should_change_lr:
                        self.reduce_lr_callback.deactivation_switch()
                    if self.should_save_model:
                        self.save_model_callback._save(
                            f'{self.save_model_callback.fname}')
                    self.set_algorithm_stage(micro_stage=4)
                # else, save the accuracy, prune useless filters and end stage
                self.accuracy_pre_pruning = max([
                    self.accuracy_FIFO[-i-1] for i in range(min(
                        self.acc_lookback, len(self.accuracy_FIFO)))])
                self.remove_filters(
                    filter_ids=useless_filters_list,
                    preserve_transition=self.preserve_transition)
                self.set_algorithm_stage(micro_stage=3)
                if self.should_change_lr:
                    self.reduce_lr_callback.activation_switch(reset_lr=True)
                # activate the countdown for keeping the kCS list as reference
                self.kCS_list_ref_cntdwn = (
                    self.m_patience_param - self.m_patience_cntdwn)
                self.set_expected_end(epoch + self.m_patience_param + 1)

            # micro-stage #3 = micro-recovery stage (accessed in next epoch)
            elif self.micro_stage == 3:
                # wait until reaching pre-pruning accuracy, then end the stage
                can_end = not self.should_change_lr or (
                    self.reduce_lr_callback.current_lr == (
                        self.reduce_lr_callback.initial_lr *
                        self.reduce_lr_callback.gamma *
                        self.reduce_lr_callback.gamma))
                if can_end and accuracy >= self.accuracy_pre_pruning:
                    # prune again if there are useless filters, else continue
                    # but first, wait for all filters to settle
                    if len(useless_filters_list) >= 1 and (
                            len(settled_filters_list) == len(self.kCS_FIFO)):
                        self.set_algorithm_stage(micro_stage=2)
                    else:
                        if self.should_change_lr:
                            self.reduce_lr_callback.deactivation_switch()
                        if self.should_save_model:
                            self.save_model_callback._save(
                                f'{self.save_model_callback.fname}')
                        self.set_algorithm_stage(micro_stage=4)
                # if the stage lasts too much time, end it
                elif self.m_re_patience_cntdwn <= 0:
                    if self.should_change_lr:
                        self.reduce_lr_callback.deactivation_switch()
                    # undo the previous pruning-recovery if saving the model
                    if self.should_save_model:
                        print("Restoring model state before previous pruning.")
                        self.add_new_filters(
                            num_new_filters=self.num_pruned_filters,
                            complementarity=False,
                            preserve_transition=self.preserve_transition)
                        print("Loading back pre-pruning weights.")
                        self.learn.load(
                            f'{self.save_model_callback.fname}_prepruning',
                            with_opt=self.save_model_callback.with_opt)
                        self.save_model_callback._save(
                            f'{self.save_model_callback.fname}')
                    self.set_algorithm_stage(micro_stage=4)
                # kCS ref and patience countdown progress
                # (at the end the kCS values remain fixed)
                self.kCS_list_ref_cntdwn -= 1
                self.m_re_patience_cntdwn -= 1

            # at the end of the micro-algorithm, try to add a new layer
            if self.micro_stage == 4:
                # reset everything for the micro-algorithm
                self.micro_stage = 0
                self.useful_filters_ceil = 0
                self.m_patience_cntdwn = self.m_patience_param
                self.m_re_patience_cntdwn = self.m_re_patience_param
                self.kCS_list_ref_cntdwn = self.m_patience_param
                self.accuracy_pre_pruning = 0
                # check if the accuracy has improved since the last layer
                # if so, add a layer, else go to the final stage
                if abs(accuracy-self.accuracy_last_layer) >= self.impr_thresh:
                    self.accuracy_last_layer = accuracy
                    self.add_new_layers(
                        preserve_transition=self.preserve_transition)
                    # alt. number of filters = half the previous
                    # layer's number if during the ascension stage.
                    #     growth_rate=floor(
                    #         len(self.filters_ref_list[-1][-1])/2))
                    self.set_algorithm_stage(micro_stage=1)
                    self.set_expected_end(epoch + 2*(self.m_patience_param+1))
                    if self.should_change_lr:
                        self.reduce_lr_callback.activation_switch(
                            reset_lr=True)
                else:
                    self.set_algorithm_stage(algorithm_stage=2)

        # stage #2 = end (stop the algorithm and reset everything)
        if self.algorithm_stage == 2:
            continue_training = False
            self.algorithm_stage = 0
            self.patience_cntdwn = self.patience_param
            self.accuracy_last_layer = 0

        return continue_training

    def self_constructing_var7(self, epoch, accuracy):
        """
        A step of DensEMANN variant #7 for one training epoch.
        Builds new layers in the last block depending on parameters.
        Returns True if training should continue, False otherwise.

        This algorithm consists of an macro-algorithm, which adds layers to the
        last block, and a micro-algorithm, which builds those layers filter by
        filter.
        - The macro-algorithm consists of a succession of two stages:
          - Improvement: call the micro-algorithm to build the last layer in
            the block (i.e. to modify it by adding/pruning filters in it);
            then check if the accuracy has improved significantly since the
            previous layer's addition (or since the beginning of the training
            if no new layers have been added yet).
            If so, adds a new layer (with the specified growth rate), and then
            either go to the ascension stage (if there is only one layer in the
            block) or start the improvement stage over (all other cases).
            Else end the algorithm.
          - Ascension: add one layer every asc_thresh training epochs; break
            the loop after std_window epochs or more if accuracy hasn't changed
            much; then go to the improvement stage.
        - The micro-algorithm consists of a succession of three stages:
            - Improvement: countdown of m_patience_param epochs; if a new
              (settled) filter becomes useful (CS above usefulness_thresh,
              automatically set), add a filter and restart the countdown;
              if the countdown ends, wait until all filters have settled and
              end the stage.
            - Pruning: save the current accuracy and prune all useless filters
              (CS below uselessness_thresh, automatically set).
            - Recovery: if/when the learning reate is at its minimal value
              (optionally after resetting the learning rate to its initial
              value and reducing it according to rlr0), wait until reaching
              pre-pruning accuracy; then if there are any new useless filters
              wait for all filters to settle and return to pruning, else end
              the stage.

        Args:
            epoch (int) - current training epoch (since adding the last block);
            accuracy (float) -  accuracy for this epoch.

        Returns:
            continue_training (bool) - whether or not the algorithm should end
                after this step.
        """
        continue_training = True

        # stage #0 = ascension stage
        if self.algorithm_stage == 0:
            if epoch == self.asc_ref_epoch:
                print("(Skipping to improvement stage.)")
                self.set_algorithm_stage(algorithm_stage=1, micro_stage=1)
                if self.should_change_lr:
                    self.reduce_lr_callback.activation_switch(reset_lr=True)
                self.set_expected_end(epoch + 2*(self.m_patience_param+1))
            else:
                # after std_window epochs in this stage, return to improvement
                # stage if the accuracy didn't change much in a while.
                if (len(self.accuracy_FIFO) >= self.std_window and
                        np.std([self.accuracy_FIFO[-i-1] for i in range(min(
                                self.std_window, len(self.accuracy_FIFO)))]) <
                        self.std_tolerance):
                    self.set_algorithm_stage(algorithm_stage=1, micro_stage=1)
                    if self.should_change_lr:
                        self.reduce_lr_callback.activation_switch(
                            reset_lr=True)
                    self.set_expected_end(epoch + 2*(self.m_patience_param+1))
                elif (epoch-self.asc_ref_epoch) % self.asc_thresh == 0:
                    self.accuracy_last_layer = accuracy
                    self.add_new_layers(
                        preserve_transition=self.preserve_transition)

        # stage #1 = improvement stage
        if self.algorithm_stage == 1:
            # This is the only stage that self-constructs at filter level,
            # so the below functionalities are only needed here
            settled_filters_list = []
            useful_filters_count = 0
            useless_filters_list = []
            kCS_settled = []
            # Update the filter kCS lists, count settled and useful filters
            kCS_list = self.learn.model.get_kCS_list_from_layer(-1, -1)
            # The actual kCS list for counting settled, useful and useless
            # filters is only updated if the associated countdown has not ended
            if self.kCS_list_ref_cntdwn > 0:
                self.kCS_list_ref = kCS_list
            for k in range(len(self.kCS_FIFO)):
                self.kCS_FIFO[k].append(self.kCS_list_ref[k])
                if len(self.kCS_FIFO[k]) == self.dkCS_smoothing:
                    self.dkCS_FIFO[k].append(
                        (self.kCS_FIFO[k][-1] - self.kCS_FIFO[k][0])/(
                            self.dkCS_smoothing-1))
                    # Settled = dkCS remained close to 0 during the last epochs
                    if ((len(self.dkCS_FIFO[k]) == self.dkCS_std_window) and (
                            np.abs(np.mean(self.dkCS_FIFO[k])
                                   ) <= self.dkCS_stl_thresh) and
                            (np.abs(np.std(self.dkCS_FIFO[k])
                                    ) <= self.dkCS_stl_thresh)):
                        settled_filters_list.append(k)
                        if self.micro_stage == 1:
                            kCS_settled.append(self.kCS_FIFO[k][-1])

            # If half of the original filters have settled
            if len(settled_filters_list) >= 0.5*self.init_num_filters:
                # During impr. stage, calculate UFT and ULT
                if self.micro_stage == 1:
                    self.usefulness_thresh = min(kCS_settled) + (
                        max(kCS_settled) - min(kCS_settled)
                        )*self.auto_usefulness_thresh
                    self.uselessness_thresh = min(kCS_settled) + (
                        max(kCS_settled) - min(kCS_settled)
                        )*self.auto_uselessness_thresh
                # Detect and count useful and useless filters
                for k in settled_filters_list:
                    # Useful = settled, and kCS above the usefulness thresh
                    if np.mean(self.kCS_FIFO[k]) >= self.usefulness_thresh:
                        useful_filters_count += 1
                    # Useless = settled, and kCS below the uselessness thresh
                    if np.mean(self.kCS_FIFO[k]) <= self.uselessness_thresh:
                        useless_filters_list.append(k)

            # micro-stage #1 = micro-improvement stage
            if self.micro_stage == 1:
                if self.m_patience_cntdwn <= 0 and (
                        len(settled_filters_list) == len(self.kCS_FIFO)):
                    # at the end of the patience countdown, end stage when all
                    # the filters have settled
                    self.set_algorithm_stage(micro_stage=2)
                elif useful_filters_count > self.useful_filters_ceil:
                    # if a new filter is useful, add a filter and restart ctdwn
                    self.useful_filters_ceil = useful_filters_count
                    self.add_new_filters(
                        num_new_filters=self.expansion_rate,
                        complementarity=self.complementarity,
                        preserve_transition=self.preserve_transition)
                    self.m_patience_cntdwn = self.m_patience_param
                    if self.should_change_lr:
                        self.reduce_lr_callback.activation_switch()
                    self.set_expected_end(epoch + 2*(self.m_patience_param+1))
                else:
                    # patience countdown progress
                    self.m_patience_cntdwn -= 1

            # micro-stage #2 = micro-pruning stage
            if self.micro_stage == 2:
                # if no filters must be pruned, or if less than
                # dont_prune_beyond filters would remain after the pruning
                # operation, skip directly to the final stage
                if len(useless_filters_list) == 0 or (
                        len(useless_filters_list) >
                        len(self.kCS_FIFO) - self.dont_prune_beyond):
                    if self.should_change_lr:
                        self.reduce_lr_callback.deactivation_switch()
                    if self.should_save_model:
                        self.save_model_callback._save(
                            f'{self.save_model_callback.fname}')
                    self.set_algorithm_stage(micro_stage=4)
                # else, save the accuracy, prune useless filters and end stage
                self.accuracy_pre_pruning = max([
                    self.accuracy_FIFO[-i-1] for i in range(min(
                        self.acc_lookback, len(self.accuracy_FIFO)))])
                self.remove_filters(
                    filter_ids=useless_filters_list,
                    preserve_transition=self.preserve_transition)
                self.set_algorithm_stage(micro_stage=3)
                if self.should_change_lr:
                    self.reduce_lr_callback.activation_switch(reset_lr=True)
                # activate the countdown for keeping the kCS list as reference
                self.kCS_list_ref_cntdwn = (
                    self.m_patience_param - self.m_patience_cntdwn)
                self.set_expected_end(epoch + self.m_patience_param + 1)

            # micro-stage #3 = micro-recovery stage (accessed in next epoch)
            elif self.micro_stage == 3:
                # wait until reaching pre-pruning accuracy, then end the stage
                can_end = not self.should_change_lr or (
                    self.reduce_lr_callback.current_lr == (
                        self.reduce_lr_callback.initial_lr *
                        self.reduce_lr_callback.gamma *
                        self.reduce_lr_callback.gamma))
                if can_end and accuracy >= self.accuracy_pre_pruning:
                    # prune again if there are useless filters, else continue
                    # but first, wait for all filters to settle
                    if len(useless_filters_list) >= 1 and (
                            len(settled_filters_list) == len(self.kCS_FIFO)):
                        self.set_algorithm_stage(micro_stage=2)
                    else:
                        if self.should_change_lr:
                            self.reduce_lr_callback.deactivation_switch()
                        if self.should_save_model:
                            self.save_model_callback._save(
                                f'{self.save_model_callback.fname}')
                        self.set_algorithm_stage(micro_stage=4)
                # if the stage lasts too much time, end it
                elif self.m_re_patience_cntdwn <= 0:
                    if self.should_change_lr:
                        self.reduce_lr_callback.deactivation_switch()
                    # undo the previous pruning-recovery if saving the model
                    if self.should_save_model:
                        print("Restoring model state before previous pruning.")
                        self.add_new_filters(
                            num_new_filters=self.num_pruned_filters,
                            complementarity=False,
                            preserve_transition=self.preserve_transition)
                        print("Loading back pre-pruning weights.")
                        self.learn.load(
                            f'{self.save_model_callback.fname}_prepruning',
                            with_opt=self.save_model_callback.with_opt)
                        self.save_model_callback._save(
                            f'{self.save_model_callback.fname}')
                    self.set_algorithm_stage(micro_stage=4)
                # kCS ref and patience countdown progress
                # (at the end the kCS values remain fixed)
                self.kCS_list_ref_cntdwn -= 1
                self.m_re_patience_cntdwn -= 1

            # at the end of the micro-algorithm, try to add a new layer
            if self.micro_stage == 4:
                # reset everything for the micro-algorithm
                self.micro_stage = 0
                self.useful_filters_ceil = 0
                self.m_patience_cntdwn = self.m_patience_param
                self.m_re_patience_cntdwn = self.m_re_patience_param
                self.kCS_list_ref_cntdwn = self.m_patience_param
                self.accuracy_pre_pruning = 0
                # check if the accuracy has improved since the last layer
                # if so, add a layer, else go to the final stage
                if abs(accuracy-self.accuracy_last_layer) >= self.impr_thresh:
                    self.accuracy_last_layer = accuracy
                    self.add_new_layers(
                        preserve_transition=self.preserve_transition)
                    # alt. number of filters = half the previous
                    # layer's number if during the ascension stage.
                    #     growth_rate=floor(
                    #         len(self.filters_ref_list[-1][-1])/2))
                    # if this is the first layer addition, adapt the DenseNet's
                    # growth rate and go to the ascension stage,
                    # else resume the current stage (improvement).
                    if self.learn.model.block_config[-1] == 2:
                        self.set_algorithm_stage(algorithm_stage=0)
                        self.asc_ref_epoch = epoch
                        self.set_expected_end(epoch + self.patience_param + 1)
                        # Also clear the accuracy_FIFO deque,
                        # as it plays an important role in the ascension stage.
                        self.accuracy_FIFO.clear()
                    else:
                        self.set_algorithm_stage(micro_stage=1)
                        self.set_expected_end(
                            epoch + 2*(self.m_patience_param+1))
                        if self.should_change_lr:
                            self.reduce_lr_callback.activation_switch(
                                reset_lr=True)
                else:
                    self.set_algorithm_stage(algorithm_stage=2)

        # stage #2 = end (stop the algorithm and reset everything)
        if self.algorithm_stage == 2:
            continue_training = False
            self.algorithm_stage = 0
            self.patience_cntdwn = self.patience_param
            self.accuracy_last_layer = 0

        return continue_training

    def self_constructing_minimal(self, epoch, accuracy):
        """
        A step of DensEMANN (minimal filter-by-filter variant)
        for one training epoch.
        Builds new layers in the last block depending on parameters.
        Returns True if training should continue, False otherwise.

        The algorithm is meant to be run with an initial architecture of 1
        layer with 1 filter. It only adds one or two additional filters to the
        layer after m_asc_thresh epochs, then ends after performing a patience
        countdown (to further train the network). This is meant to represent a
        "minimal" filter-level self-construction.

        Args:
            epoch (int) - current training epoch (since adding the last block);
            accuracy (float) -  accuracy for this epoch.

        Returns:
            continue_training (bool) - whether or not the algorithm should end
                after this step.
        """
        continue_training = True
        max_filters = 3

        # Update the filter kCS lists
        kCS_list = self.learn.model.get_kCS_list_from_layer(-1, -1)
        for k in range(len(self.kCS_FIFO)):
            self.kCS_FIFO[k].append(kCS_list[k])

        # stage #0 = ascension stage (currently does nothing)
        if self.algorithm_stage == 0:
            print("(This variant hasn't got an ascension stage,"
                  " skipping to improvement.)")
            self.set_algorithm_stage(algorithm_stage=1, micro_stage=0)

        # stage #1 = improvement stage
        if self.algorithm_stage == 1:
            # micro-stage #0 = minimal micro-ascension stage
            if self.micro_stage == 0:
                if len(kCS_list) >= max_filters:
                    # end stage when there are at least max_filters filters
                    self.set_algorithm_stage(micro_stage=3)
                    if self.should_change_lr:
                        self.reduce_lr_callback.activation_switch(
                            reset_lr=True)
                    self.set_expected_end(epoch + self.m_patience_param + 1)
                elif (epoch-self.asc_ref_epoch+1) % self.m_asc_thresh == 0:
                    self.add_new_filters(
                        complementarity=self.complementarity,
                        preserve_transition=self.preserve_transition)

            # micro-stage #3 = micro-recovery stage
            if self.micro_stage == 3:
                if self.m_patience_cntdwn <= 0:
                    self.set_algorithm_stage(micro_stage=4)
                else:
                    self.m_patience_cntdwn -= 1

            # micro-stage #4 = end
            if self.micro_stage == 4:
                # reset everything for the micro-algorithm
                self.micro_stage = 0
                self.useful_filters_ceil = 0
                self.m_patience_cntdwn = self.m_patience_param
                self.set_algorithm_stage(algorithm_stage=2)

        # stage #2 = end (stop the algorithm and reset everything)
        if self.algorithm_stage == 2:
            continue_training = False
            self.algorithm_stage = 0

        return continue_training

    def after_epoch(self):
        """
        After each training epoch, perform a step of the DensEMANN algorithm.
        """
        if self.active:
            # Get the last recorded accuracy on the validation set,
            # and save it to the FIFO list.
            accuracy = next(m for m in self.recorder._valid_mets
                            if m.name == "accuracy").value.item()
            self.accuracy_FIFO.append(accuracy)
            # Get the current epoch.
            epoch = self.learn.epoch
            # Perform a step of the DensEMANN algorithm.
            continue_training = self.self_constructing_step(epoch, accuracy)
            # Stop DensEMANN when specified by the algorithm.
            if not continue_training:
                # If the block_count hasn't yet been reached, add a new block
                # and resume DensEMANN. Else end the training process.
                if len(self.learn.model.block_config) < self.block_count:
                    self.add_new_block()
                    self.initialise_algorithm_variables(asc_ref_epoch=epoch)
                else:
                    self.active = False
                    raise CancelFitException()


class ReduceLRCallback(Callback):
    """
    Custom callback for learning rate (LR) scheduling.
    The schedule consists in multiplying the LR by a 'gamma' parameter at two
    milestones 'rlr_1' and 'rlr_2', corresponding to two fractions of a period
    of 'schedule_length' epochs (which by default corresponds to the total
    number of training epochs).
    If 'self_construct_mode' is set, the LR schedule is deactivated, and may be
    re-activated externally at any moment. In that case, the milestones
    correspond to fractions of a period of 'schedule_length' epochs begining
    after the epoch in which the schedule is re-activated.

    Args:
        rlr_var (int) - the variant of the learning rate modification method
            to implement, mainly related to what to do after external
            activation (default 0).
        lr (float) - initial learning rate (default 0.1).
        gamma (float) - multiplicative factor for scheduled LR modifications
            (default 0.1, i.e. a division by 10).
        rlr_1 (float) - first scheduling milestone for multiplying the LR by
            gamma (default 0.5, i.e. 50% through the training process).
        rlr_2 (float) - second scheduling milestone for multiplying the LR by
            gamma (default 0.75, i.e. 75% through the training process).
        schedule_length (int or None) - length of the LR schedule in training
            epochs, with a None value interpreted as 'the total number of
            training epochs' (default None).
        self_construct_mode (bool) - whether or not to initialize the callback
            with the LR schedule deactivated.

    Attributes:
        rlr_var (int) - from args.
        initial_lr (float) - initial LR value, saved as a reference.
        current_lr (float) - current LR value, changed by scheduled actions.
        gamma (float) - from args.
        rlr_1 (float) - from args.
        rlr_2 (float) - from args.
        first_epoch (int) - epoch at which the LR schedule begins (0 if not in
            self_construct_mode, else it is the epoch in which the LR
            schedule is re-activated).
        schedule_length (int) - from args (with None values interpreted as
            specified in args).
        active (bool) - whether or not the LR schedule is active.
    """
    order = 75

    def __init__(self, rlr_var=0, lr=0.1, gamma=0.1, rlr_1=0.5, rlr_2=0.75,
                 schedule_length=None, self_construct_mode=False):
        """
        Initializer for the ReduceLRCallback.
        """
        self.rlr_var = rlr_var
        self.initial_lr = lr
        self.current_lr = lr
        self.gamma = gamma
        self.rlr_1 = rlr_1
        self.rlr_2 = rlr_2
        self.first_epoch = 0
        self.schedule_length = schedule_length
        # In self_construct_mode, the schedule is deactivated.
        self.active = not self_construct_mode

    def activation_switch(self, reset_lr=False):
        """
        Activate the LR schedule, with the current epoch set as its begining.
        The effect on a previous or ongoing LR modification depends on rlr_var.
        In variant 0, the LR returns to its initial value after the activation.
        In variant 1, the LR remains unchanged.
        A reset of the LR to its initial value may also be forced manually.

        Args:
            reset_lr (bool) - whether or not the LR should be reset to its
                initial value, regardless of rlr_var (default False).
        """
        self.active = True
        self.first_epoch = self.learn.epoch
        if self.rlr_var == 0 or reset_lr:
            self.current_lr = self.initial_lr

    def deactivation_switch(self):
        """
        Deactivate the LR schedule, and set the LR to its final value.
        """
        self.active = False
        self.current_lr = self.initial_lr * self.gamma * self.gamma

    def before_fit(self):
        """
        Before the training process begins, if the schedule_length value is
        None, it is replaced by total number of epochs.
        """
        if self.schedule_length is None:
            self.schedule_length = self.learn.n_epoch

    def before_batch(self):
        """
        At every batch, the learning rate is updated.
        """
        self.opt.set_hyper('lr', self.current_lr)

    def after_train(self):
        """
        Main action of the ReduceLRCallback (takes place after the training
        phase). Performs the scheduled LR modification.
        """
        if self.active:
            # The LR modification is performed at milestones rlr_1 and rlr_2
            # between self.first_epoch and self.schedule_length.
            # It is also only performed once (unless the LR has been reset).
            rlr_1_epoch = int(self.schedule_length * self.rlr_1)
            rlr_2_epoch = int(self.schedule_length * self.rlr_2)
            if ((self.learn.epoch - self.first_epoch == rlr_1_epoch) and (
                    self.current_lr == self.initial_lr)) or ((
                    self.learn.epoch - self.first_epoch == rlr_2_epoch) and (
                    self.current_lr == self.initial_lr * self.gamma)):
                self.current_lr = self.current_lr * self.gamma
                print("LR has been multplied by %f, new LR = %f" % (
                    self.gamma, self.current_lr))
            # After schedule_length, there is no need to keep this active.
            elif self.learn.epoch - self.first_epoch == self.schedule_length:
                self.active = False


class CSVLoggerCustom(Callback):
    """
    Modification of the fastai CSVLogger callback.
    (https://github.com/fastai/fastai/blob/
    6354225a2a275026ee8e69e9977c050ebe6ed008/fastai/callback/progress.py#L93)
    Log the results displayed in 'learn.path/fname', using special chars for
    commas (separators) and decimals as specified by the user.

    Args:
        fname (str) - the file name for the CSV log (default 'history.csv').
        add_ft_kCS (bool) - whether or not to add the kCS values from filters
            in each layer to the CSV log.
        ft_comma (str) - 'comma' separator in the CSV logs (default ';').
        ft_decimal (str) - 'decimal' separator in the CSV logs (default ',').

    Attributes:
        append (bool) - whether or not to append lines to an existing CSV log
            (if the file exists, lines are appended to it).
        fname (str) - from args.
        add_ft_kCS (bool) - from args.
        ft_comma (str) - from args.
        ft_decimal (str) - from args.
    """
    order = 61

    def __init__(self, fname='history.csv',  add_ft_kCS=True,
                 ft_comma=';', ft_decimal=','):
        """
        Initializer for the CSVLoggerCustom callback.
        """
        self.fname = Path(fname)
        self.append = os.path.isfile(self.fname)
        self.add_ft_kCS = add_ft_kCS
        self.ft_comma = ft_comma
        self.ft_decimal = ft_decimal

    def read_log(self):
        """
        Convenience method to quickly access the log.
        """
        return pd.read_csv(self.path/self.fname)

    def before_fit(self):
        """
        Prepare file with metric names.
        """
        if hasattr(self, "gather_preds"):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = (self.path/self.fname).open('a' if self.append else 'w')
        self.file.write(self.ft_comma.join(self.recorder.metric_names))
        if self.add_ft_kCS:
            self.file.write(2*self.ft_comma + 'kCS for each layer\n')
        else:
            self.file.write('\n')
        self.old_logger, self.learn.logger = self.logger, self._write_line

    def _write_line(self, log):
        """
        Write a line with 'log' (and optionally also the kCS),
        and call the old logger.
        """
        # Features in 'log'.
        self.file.write(self.ft_comma.join(
            [str(t).replace('.', self.ft_decimal) for t in log]))
        # kCS for each layer.
        if self.add_ft_kCS:
            for block in range(len(self.learn.model.block_config)):
                for layer in range(self.learn.model.block_config[block]):
                    kCS_list = self.learn.model.get_kCS_list_from_layer(
                        block, layer)
                    self.file.write(2*self.ft_comma + self.ft_comma.join(
                        [str(kCS).replace(
                            '.', self.ft_decimal) for kCS in kCS_list]))
                if block != len(self.learn.model.block_config) - 1:
                    self.file.write(self.ft_comma)
        self.file.write('\n')
        self.file.flush()
        os.fsync(self.file.fileno())
        self.old_logger(log)

    def after_fit(self):
        """
        Close the file and clean up.
        """
        if hasattr(self, "gather_preds"):
            return
        self.file.close()
        self.learn.logger = self.old_logger
