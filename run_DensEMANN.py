# Executable for PyTorch + fastai DensEMANN implementation.
# Based on:
# - TensorFlow DensEMANN implementation by me:
#   https://github.com/AntonioGarciaDiaz/Self-constructing_DenseNet
# - PyTorch efficient DenseNet-BC implementation by Geoff Pleiss:
#   https://github.com/gpleiss/efficient_densenet_pytorch
# - TensorFlow DenseNet implementation by Illarion Khlestov:
#   https://github.com/ikhlestov/vision_networks

import argparse
import os
from controllers import DensEMANN_controller

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # What actions (train or test) to do with the model.
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model.')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for specified dataset if pretrained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.set_defaults(train=False)
    parser.set_defaults(test=False)

    # PARAMETERS RELATED TO LOADING AND SAVING FILES --------------------------
    # -------------------------------------------------------------------------
    # In case an existing model should be reused.
    parser.add_argument(
        '--source_experiment_id', '--source_id', default=None,
        dest='source_experiment_id', type=str,
        help='Experiment ID for the existing model to load (default None, i.e.'
             ' do not load a source experiment).')
    parser.add_argument(
        '--import-trainable-values-and-architecture',
        '--import-weights-and-architecture',
        '--import-trainable-values-and-hyperparameters',
        '--import-weights-and-hyperparameters',
        '--import-trainable-values-and-hypers', '--import-weights-and-hypers',
        '--import-trainable-values', '--import-weights',
        dest='import_weights', action='store_true',
        help='Import the weights and trainable values from the existing model'
             ' (as specified in the model file) as well as the architecture'
             ' (default option).')
    parser.add_argument(
        '--import-only-architecture', '--import-only-hyperparameters',
        '--import-only-hypers', '--no-import-trainable-values',
        '--no-import-weights',
        dest='import_weights', action='store_false',
        help='Do not import the weights and trainable values from the'
             ' existing model (i.e. only import the architecture).')
    parser.set_defaults(import_weights=True)
    parser.add_argument(
        '--reuse-existing-files', '--reuse-files',
        dest='reuse_files', action='store_true',
        help='Keep using the same files (logs, model) as existing model.')
    parser.add_argument(
        '--no-reuse-existing-files', '--no-reuse-files',
        dest='reuse_files', action='store_false',
        help='Create new files, do not reuse files for existing model.'
             ' (default option).')
    parser.set_defaults(reuse_files=False)

    # Paths for loading and saving data.
    parser.add_argument(
        '--data', dest='data', type=str, default=None,
        help='Path to directory where data should be loaded from/downloaded'
             ' (default None, i.e. a new folder at the tempfile).')
    parser.add_argument(
        '--load', dest='load', type=str, default=None,
        help='Path to directory where a source model should be loaded from'
             ' (default None, i.e. a \'ft-logs\' folder in the current working'
             ' directory).')
    parser.add_argument(
        '--save', dest='save', type=str, default=None,
        help='Path to directory where the model and ft-logs should be saved to'
             ' (default None, i.e. a \'ft-logs\' folder in the current working'
             ' directory). Is overwritten with the --load value if'
             ' --reuse-existing-files.')

    # GENERAL EXECUTION PARAMETERS --------------------------------------------
    # -------------------------------------------------------------------------
    # Hyperarameters that define the current DenseNet model.
    parser.add_argument(
        '--growth_rate', '-k', dest='growth_rate', type=int,
        default=12,  # choices in paper: 12, 24, 40.
        help='Growth rate: number of features added per DenseNet layer,'
             ' corresponding to the number of convolution filters in that'
             ' layer (default 12).')
    parser.add_argument(
        '--layer_num_list', '-lnl', dest='layer_num_list',
        type=str, default='1',
        help='The block configuration in string form: list of the (initial)'
             ' number of convolution layers in each block, separated by'
             ' commas (e.g. \'12,12,12\')'
             ' (default: \'1\', i.e. 1 block with 1 layer)'
             ' WARNING: in BC models, each layer is preceded by a bottleneck.')
    parser.add_argument(
        '--filter_num_list', '-fnl', dest='filter_num_list',
        type=str, default=None,
        help='The block configuration in string form, with an extra level of'
             ' detail: list of the (initial) number of convolution filters in'
             ' each layer, in each block. Blocks are separated by dot commas,'
             ' while layers are separated by commas. For instance:'
             ' \'12,12,12;13,13,13\' means 2 blocks with 3 layers each, 12'
             ' filters in each layer for the first block, and 13 filters in'
             ' each layer for the second block'
             ' (unused by default, overrides layer_num_list if used).'
             ' WARNING: in BC models, each layer is preceded by a bottleneck;'
             ' the number of filters in it is managed by update_growth_rate.')
    parser.add_argument(
       '--update-growth-rate-each-layer', '--update-growth-rate',
       '--update-k-each-layer', '--update-k',
       dest='update_growth_rate', action='store_true',
       help='Update the DenseNet\'s default growth rate value before each'
            ' layer or block addition: the new value will correspond to the'
            ' number of filters in the previous layer. For DenseNet-BC, this'
            ' arg also means that, when the model is created, the size of each'
            ' bottleneck layer is always 4 x the number of filters in the'
            ' previous convolutional layer (useful if using filter_num_list).')
    parser.add_argument(
       '--same-growth-rate-each-layer', '--same-growth-rate',
       '--same-k-each-layer', '--same-k',
       '--no-update-growth-rate-each-layer', '--no-update-growth-rate',
       '--no-update-k-each-layer', '--no-update-k',
       dest='update_growth_rate', action='store_false',
       help='Do not update the DenseNet\'s default growth rate value'
            ' (default option).  For DenseNet-BC, this arg also means that,'
            ' when the model is created, the size of each bottleneck layer is'
            ' always 4 x the specified growth_rate value.')
    parser.set_defaults(update_growth_rate=False)
    parser.add_argument(
        '--keep_prob', '-kp', dest='keep_prob', type=float,
        help='Keep probability for dropout, if keep_prob = 1 dropout will be'
             ' disabled (default 1.0 with data augmentation, 0.8 without it).')
    parser.add_argument(
        '--model_type', '-m', dest='model_type', type=str,
        choices=['DenseNet', 'DenseNet-BC'], default='DenseNet-BC',
        help='Model type name: \'DenseNet\' or \'DenseNet-BC\', where'
             ' \'DenseNetBC\' uses bottleneck + compression'
             ' (default \'DenseNet-BC\').')
    parser.add_argument(
        '--dataset', '-ds', dest='dataset', type=str,
        choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN',
                 'ImageNet', 'ImageNet+',
                 'FMNIST', 'FMNIST+', 'FER2013', 'FER2013+'], default='C10+',
        help='Dataset name, if followed by a \'+\' data augmentation is added'
             ' (default \'C10+\', i.e. CIFAR-10 with data augmentation)')
    parser.add_argument(
        '--resize', dest='resize', type=int, default=None,
        help='Resize the images in the dataset to a user-specified value'
             ' (default None, i.e. do not resize the images).')
    parser.add_argument(
        '--cutout', dest='cutout', action='store_true',
        help='Use CutOut data augmentation from DeVries and Taylor 2017'
             ' (independent from standard \'+\' data augmentation).')
    parser.add_argument(
        '--no-cutout', dest='cutout', action='store_false',
        help='Do not use CutOut data augmentation.')
    parser.set_defaults(cutout=False)
    parser.set_defaults(reuse_files=False)
    parser.add_argument(
        '--reduction', '-red', '-theta', dest='reduction',
        type=float, default=0.5,
        help='Reduction (theta) at transition layer for DenseNets with'
             ' compression (DenseNets-BC models) (default 0.5).')

    # Other training-related parameters.
    parser.add_argument(
        '--efficient', dest='efficient', action='store_true',
        help='Use the implementation with checkpointing'
             ' (slow but memory efficient), which is disabled by default.')
    parser.set_defaults(efficient=False)
    parser.add_argument(
        '--training_set_size', '--training_size',
        '--train_set_size', '--train_size',
        dest='train_size', type=int,
        help='Size of the training set, number of examples cut off the'
             ' original training set for training.'
             ' (default 45000 for C10 and C100, 6000 for SVHN).'
             ' The max value for this argument corresponds to the size of the'
             ' original training set - the size of the validation set.'
             ' This max value is used instead of the specified value if the'
             ' latter exceeds the former.')
    parser.add_argument(
        '--validation_set_size', '--validation_size',
        '--valid_set_size', '--valid_size',
        dest='valid_size', type=int,
        help='Size of the validation set, number of examples cut off the'
             ' original training set for validation'
             ' (default 5000 for C10 and C100, 6000 for SVHN).')
    parser.add_argument(
        '--num_epochs', '--n_epochs', '-nep',
        dest='n_epochs', type=int, default=300,
        help='Number of epochs for training, used when not self-constructing'
             ' and for DensEMANN variants 0 and 1 (default 300).')
    parser.add_argument(
        '--limit_num_epochs', '--lim_n_epochs', '-lnep',
        dest='lim_n_epochs', type=int, default=99999999,
        help='Upper limit to the number of training epochs performed when'
             ' self-constructing, used for most variants of DensEMANN'
             ' (default 99999999).')
    parser.add_argument(
        '--batch_size', '--b_size', '-bs',
        dest='batch_size', type=int, default=64,
        help='Number of images in a training batch (default 64).')
    parser.add_argument(
        '--learning_rate', '-lr',
        dest='lr', type=float, default=0.1,
        help='Initial learning rate (LR) (default 0.1).')
    parser.add_argument(
        '--reduce_learning_rate_gamma', '--rlr_gamma', '-gamma',
        dest='gamma', type=float, default=0.1,
        help='Multiplicative factor for scheduled LR modifications'
             ' (default 0.1, i.e. a division by 10).')
    parser.add_argument(
        '--reduce_learning_rate_1', '--rlr_1', '-rlr1',
        dest='rlr_1', type=float, default=0.5,
        help='First scheduling milestone for multiplying the LR by gamma,'
             ' if following DensEMANN\'s LR modification schedule'
             ' (default 0.5, i.e. 50% through the training process).')
    parser.add_argument(
        '--reduce_learning_rate_2', '--rlr_2', '-rlr2',
        dest='rlr_2', type=float, default=0.75,
        help='Second scheduling milestone for multiplying the LR by gamma,'
             ' if following DensEMANN\'s LR modification schedule'
             ' (default 0.75, i.e. 75% through the training process).')
    parser.add_argument(
        '--weight_decay', '-wd', dest='wd', type=float, default=1e-4,
        help='Weight decay for the loss function (default 0.0001).')
    parser.add_argument(
        '--nesterov_momentum', '--momentum', '-nm',
        dest='momentum', type=float, default=0.9,
        help='Nesterov momentum for the optimizer (default 0.9).')
    parser.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help='Optional seed for the random number generator (default None)'
             ' WARNING: Currently doesn\'t produce true determinism.')

    # What kind of algorithm (self-constructing, training, etc.) to apply.
    parser.add_argument(
        '--DensEMANN', '--self-construct', '--self-constr',
        dest='should_self_construct', action='store_true',
        help='Use the DensEMANN self-constructing algorithm to modify the'
             ' network\'s initial architecture during training'
             ' (default option).')
    parser.add_argument(
        '--prebuilt', '--no-self-construct', '--no-self-constr',
        dest='should_self_construct', action='store_false',
        help='Do not use DensEMANN, only train the specified (prebuilt)'
             ' architecture.')
    parser.set_defaults(should_self_construct=True)
    parser.add_argument(
        '--pruning', '--prune', '--sparsification', '--sparsify', '--spars',
        dest='should_sparsify', action='store_true',
        help='If not using DensEMANN, run a scheduled sparsification on the'
             ' network (zero-out learnable parameters to \'prune\' them).')
    parser.add_argument(
        '--no-pruning', '--no-prune', '--no-sparsification', '--no-sparsify',
        '--no-spars', dest='should_sparsify', action='store_false',
        help='Do not run a scheduled sparsification on the network'
             ' (default option).')
    parser.set_defaults(should_sparsify=False)
    parser.add_argument(
        '--change-learning-rate', '--change-lr',
        dest='should_change_lr', action='store_true',
        help='Allow any modifications to the LR as defined in the training'
             ' algorithm (default option). When not using DensEMANN, the LR'
             ' schedule is specified using the --prebuilt_rlr argument.')
    parser.add_argument(
        '--no-change-learning-rate', '--no-change-lr',
        dest='should_change_lr', action='store_false',
        help='Do not allow any modifications to the LR, regardless of the'
             ' training algorithm.')
    parser.set_defaults(should_change_lr=True)

    # DensEMANN-RELATED PARAMETERS --------------------------------------------
    # -------------------------------------------------------------------------

    # Parameters that define the DensEMANN variant to use.
    parser.add_argument(
        '--DensEMANN_variant', '--DensEMANN_var',
        '--self_constructing_variant', '--self_constructing_var',
        '--self_constr_variant', '--self_constr_var', '-var',
        dest='self_constructing_var', type=int, default=-1,
        help='Choice on the DensEMANN variant to use (from oldest to newest).'
             ' Variants are identified by an int value (0, 1, 2, 3, etc.).'
             ' They are described in the DensEMANNCallback, in their'
             ' respective functions (self_constructing_varX).'
             ' Passing a negative value, or one that does not (yet) identify a'
             ' variant, results in running the most recent operational variant'
             ' (default -1, i.e. most recent variant).')
    parser.add_argument(
        '--DensEMANN_reduce_lr', '--DensEMANN_rlr',
        '--self_constructing_reduce_lr', '--self_constructing_rlr',
        '--self_constr_reduce_lr', '--self_constr_rlr', '-rlr',
        dest='self_constr_rlr', type=int, default=0,
        help='Choice on the learning rate reduction variant to use with the'
             ' DensEMANN algorithm (from oldest to newest).'
             ' Variants are identified by an int value (0, 1).'
             ' They are described in the ReduceLRCallback, in the'
             ' activation_switch function (where they take effect).'
             ' Passing a negative value, or one that does not (yet) identify a'
             ' variant, results in running the most recent operational variant'
             ' (default 0).')

    # Layer-level DensEMANN parameters.
    parser.add_argument(
        '--block_count', '--blocks', '-bc',
        dest='block_count', type=int, default=1,
        help='Minimum number of dense blocks in the network for ending the'
             ' self-construction process. If, at the end of DensEMANN, the'
             ' number of blocks in the network is lower than block_count,'
             ' new blocks are created through the method specified by'
             ' --new_block_mode (default=1).')
    parser.add_argument(
        '--new_block_mode', '--nblock_mode', '--nbm',
        dest='new_block_mode', type=str,
        choices=['from_scratch', 'brutal_copy',
                 'incremental_copy', 'reset_copy'], default='brutal_copy',
        help='Mode for adding new blocks. The choice is between'
             ' \'from_scratch\' (use DensEMANN to build each new block),'
             ' \'brutal_copy\' (default, add all the new blocks at once as'
             ' copies of the first one, and then train for --patience_param'
             ' epochs), \'incremental_copy\' (add the new blocks one by'
             ' one as copies of the first one, training for --patience_param'
             ' epochs after each block addition), and'
             ' \'reset_copy\' (add all the new blocks as copies of the'
             ' first one, reinitialize the network\'s weight values, and'
             ' then train for --patience_param epochs).')
    parser.add_argument(
        '--layer_connection_strength', '--layer_cs', '-lcs', dest='layer_cs',
        type=str, choices=['relevance', 'spread'], default='relevance',
        help='Choice on \'layer CS\', how to interpret connection strength'
             ' (CS) data for layers in DensEMANN variants 0 to 3.'
             ' Relevance (default) evaluates a given layer\'s connections from'
             ' the perspective of other layers. Spread evaluates them from the'
             ' perspective of that given layer.')
    parser.add_argument(
        '--ascension_threshold', '--asc_thresh', '-at',
        dest='asc_thresh', type=int, default=10,
        help='Ascension threshold, used for self-constructing at layer level'
             ' in DensEMANN variants 0 to 3 and 7. Number of epochs before'
             ' adding a new layer during the ascension stage. (default=10).')
    parser.add_argument(
        '--patience_parameter', '--patience_param', '-pp',
        dest='patience_param', type=int, default=300,
        help='Patience parameter, used for self-constructing at layer level'
             ' in DensEMANN variants 0 to 3. Number of epochs to wait before'
             ' stopping the improvement stage, unless a new layer settles.'
             ' (default=200).')
    parser.add_argument(
        '--accuracy_std_tolerance', '--std_tolerance', '-stdt',
        dest='std_tolerance', type=float, default=0.1,
        help='Accuracy St.D. tolerance, used for self-constructing at layer'
             ' level in DensEMANN variants 2, 3 and 7.'
             ' Minimum standard deviation value for a window of previous'
             ' accuracy values in the ascension stage. If the St.D. of the'
             ' previous accuracies (std_window) goes below std_tolerance,'
             ' the ascension stage ends (default 0.1).')
    parser.add_argument(
        '--accuracy_std_window', '--acc_std_window', '--std_window', '-stdw',
        dest='std_window', type=int, default=50,
        help='Accuracy St.D. window, used for self-constructing at layer'
             ' level in DensEMANN variants 2, 3 and 7.'
             ' Number of previous accuracy values that are taken into account'
             ' for deciding if the ascension stage should end'
             ' (this happens when the St.D. of these accuracy values is below'
             ' std_tolerance) (default 50).')
    parser.add_argument(
        '--accuracy_improvement_threshold', '--accuracy_impr_thresh',
        '--acc_improvement_threshold', '--acc_impr_thresh',
        '--improvement_threshold', '--impr_thresh', '-it',
        dest='impr_thresh', type=float, default=0.01,
        help='Accuracy improvement threshold, used for self-constructing at'
             ' layer level in DensEMANN variants 4 onwards. Minimum absolute'
             ' difference between the accuracy after the completion of the'
             ' last and before-last layer during the improvement stage. If the'
             ' difference between the two accuracies is below the improvement'
             ' threshold, the improvement stage ends (default 0.01).')
    parser.add_argument(
        '--preserve-transition-each-layer', '--preserve-transition',
        dest='preserve_transition', action='store_true',
        help='Preserve parts of the previous final transition layer every time'
             ' a layer is added (within the same block) (default option).')
    parser.add_argument(
        '--new-transition-each-layer', '--new-transition',
        '--no-preserve-transition-each-layer', '--no-preserve-transition',
        dest='preserve_transition', action='store_false',
        help='Create a new final transition layer every time a layer is added'
             ' (within the same block).')
    parser.set_defaults(preserve_transition=True)
    parser.add_argument(
        '--undo-last-layer-addition', '--undo-last-layer',
        '--remove-last-layer-at-end', '--remove-last-layer',
        dest='remove_last_layer', action='store_true',
        help='After building a dense block, undo its last layer\'s addition'
             ' (remove the last layer and reload the weight values before the'
             ' layer was added) (default option). Implies should_save_model is'
             ' set to true (via --model-saves).')
    parser.add_argument(
        '--keep-last-layer-at-end', '--keep-last-layer',
        '--no-undo-last-layer-addition', '--no-undo-last-layer',
        '--no-remove-last-layer-at-end', '--no-remove-last-layer',
        dest='remove_last_layer', action='store_false',
        help='After building a dense block, keep its last added layer'
             ' (i.e. do not remove it / undo its addition).')
    parser.set_defaults(remove_last_layer=True)

    # Filter-level DensEMANN parameters.
    parser.add_argument(
        '--expansion_rate', '-fex', type=int,
        default=1,  # by default, new filters are added one by one
        help='Expansion rate (rate at which new convolution filters are added'
             ' together during the self-construction of a dense layer)'
             ' (default 1).')
    parser.add_argument(
        '--derivate_filter_CS_smoothing', '--der_filter_CS_smoothing',
        '--der_kCS_smoothing', '--dkCS_smoothing',
        '--der_kCS_smooth', '-dkCS_smooth',
        dest='dkCS_smoothing', type=int, default=10,
        help='Smoothing when calculating the derivate of each filter\'s kCS'
             ' (number of epochs to look back when calculating the slope).'
             ' Used for self-constructing at filter level, to know which'
             ' filters have settled (default 10).')
    parser.add_argument(
        '--derivate_filter_CS_std_window', '--der_filter_CS_std_window',
        '--der_kCS_std_window', '--dkCS_std_window',
        '--der_kCS_stdw', '-dkCS_stdw',
        dest='dkCS_std_window', type=int, default=30,
        help='St.D. window for the for the derivate of each filter\'s kCS'
             ' (number of epochs to look back to calculate the St.D. of the'
             ' kCS derivate). Used for self-constructing at filter level,'
             ' to know which filters have settled (default 30).')
    parser.add_argument(
        '--derivate_filter_CS_settling_threshold',
        '--derivate_filter_CS_stl_thresh',
        '--der_filter_CS_settling_threshold', '--der_filter_CS_stl_thresh',
        '--der_kCS_stl_thresh', '--dkCS_stl_thresh',
        '--der_kCS_st', '-dkCS_st',
        dest='dkCS_stl_thresh', type=float, default=0.001,
        help='Settling threshold for filter kCS derivates'
             ' (i.e. a value which means \'close to 0\' for them).'
             ' Used for self-constructing at filter level, to know which'
             ' filters have settled (default 0.001).')
    parser.add_argument(
        '--auto_usefulness_threshold', '--auto_usefulness_thresh', '-auto_uft',
        '--usefulness_threshold_auto', '--usefulness_thresh_auto', '-uft_auto',
        '--usefulness_threshold', '--usefulness_thresh', '-uft',
        dest='auto_usefulness_thresh', type=float, default=0.8,
        help='Usefulness threshold for filter CS (i.e. the kCS value above'
             ' which it is considered that the filter is transmitting'
             ' \'useful\' information), expressed as a percentage.'
             ' Used for automatically setting the usefulness threshold: after'
             ' k/2 filters have settled, it is found between the highest and'
             ' lowest kCS of settled filters at that fraction of the distance'
             ' between them (default 0.8).')
    parser.add_argument(
        '--auto_uselessness_threshold', '--auto_uselessness_thresh',
        '-auto_ult', '--uselessness_threshold_auto',
        '--uselessness_thresh_auto', '-ult_auto',
        '--uselessness_threshold', '--uselessness_thresh', '-ult',
        dest='auto_uselessness_thresh', type=float, default=0.2,
        help='Uselessness threshold for filter CS (i.e. the kCS value below'
             ' which it is considered that the filter is NOT transmitting any'
             ' \'useful\' information), expressed as a percentage .'
             ' Used for automatically setting the uselessness threshold: after'
             ' k/2 filters have settled, it is found between the highest and'
             ' lowest kCS of settled filters at that fraction of the distance'
             ' between them (default 0.2).')
    parser.add_argument(
        '--micro_ascension_threshold', '--m_asc_thresh', '-mat',
        dest='m_asc_thresh', type=int, default=5,
        help='Micro-ascension threshold, used for self-constructing at filter'
             ' level in DensEMANN variants 4 onwards. Number of epochs before'
             ' adding a new filter during the micro-ascension stage, until a'
             ' filter settles (default 5).')
    parser.add_argument(
        '--micro_improvement_patience_parameter',
        '--m_improvement_patience_param',
        '--micro_impr_patience_parameter', '--m_impr_patience_param', '-mipp',
        '--micro_patience_parameter', '--m_patience_param', '-mpp',
        dest='m_patience_param', type=int, default=300,
        help='Micro-patience threshold, used for self-constructing at filter'
             ' level in DensEMANN variants 4 onwards. Number of epochs to wait'
             ' before stopping the micro-improvement stage, unless a new'
             ' filter settles (default 300).')
    parser.add_argument(
        '--filter-complementarity', '--complementarity',
        '--filter-compl', '--compl',
        dest='complementarity', action='store_true',
        help='Use a complementarity mechanism when adding new filters: the'
             ' sign configuration is the opposite of that of the filters with'
             ' lowest CS, unless that configuration already exists in the'
             ' layer (in which case it must be differnet but close)'
             ' (default option).')
    parser.add_argument(
        '--filter-random-initialisation', '--random-initialisation',
        '--filter-random-init', '--random-init', '--filter-random',
        '--no-filter-complementarity', '--no-complementarity',
        '--no-filter-compl', '--no-compl',
        dest='complementarity', action='store_false',
        help='Do not use a complementarity mechanism when adding new filters.')
    parser.set_defaults(complementarity=True)
    parser.add_argument(
        '--dont_prune_beyond', '-dpb',
        dest='dont_prune_beyond', type=int, default=1,
        help='Minimum number of filters that should remain after a pruning'
             ' operation. If more filters have to be pruned, the algorithm'
             ' stops building the layer (default 1, i.e. the algorithm must'
             ' spare at least 1 filter in the layer).')
    parser.add_argument(
        '--micro_recovery_patience_parameter', '--m_recovery_patience_param',
        '--micro_re_patience_parameter', '--m_re_patience_param', '-mrpp',
        dest='m_re_patience_param', type=int, default=1000,
        help='Alternate micro-patience threshold, used for self-constructing'
             ' at filter level in DensEMANN variants 4 onwards. Number of'
             ' epochs to wait before terminating the micro-recovery stage'
             ' if pre-pruning accuracy (the usual condition for stopping this'
             ' stage) cannot be reached (default 1000).')

    # PARAMETERS USED FOR PROCESSES NOT RRELATED TO DensEMANN -----------------
    # -------------------------------------------------------------------------

    # General sparsification-related parameters.
    parser.add_argument(
        '--end_sparsity', '--end_spars',
        dest='end_sparsity', type=float, default=50,
        help='Value between 0 and 100, corresponding to the percentage of the'
             ' trainable parameters that should be zeroed-out during scheduled'
             ' sparsification (default 50).')
    parser.add_argument(
        '--sparsifier_granularity', '--spars_granularity', '--granularity',
        dest='spars_granularity', type=str,
        choices=['weight', 'shared_weight', 'column', 'row', 'channel',
                 'kernel', 'filter'], default='filter',
        help='Granularity for sparsification. The choice is between 0D'
             ' \'weight\' level (with the possibility of applying the same'
             ' pattern to all filters via \'shared_weight\'), 1D \'column\','
             ' \'row\' and \'channel\' levels, 2D \'kernel\' level, and 3D'
             ' \'filter\' level (default). More info at:'
             ' https://nathanhubens.github.io/fasterai/sparsifier.html')
    parser.add_argument(
        '--sparsifier_method', '--spars_method', '--method',
        dest='spars_method', type=str,
        choices=['local', 'global'], default='global',
        help='Method for sparsification: either \'local\' (sparsify each layer'
             ' separately), or \'global\' (default, sparsify the network as a'
             ' whole).')
    parser.add_argument(
        '--sparsifier_schedule_function', '--sparsifier_schedule_func',
        '--sparsifier_sched_function', '--sparsifier_sched_func',
        '--spars_schedule_function', '--spars_schedule_func',
        '--spars_sched_function', '--spars_sched_func',
        dest='spars_sched_func', type=str,
        choices=['one_shot', 'iterative', 'sched_agp', 'sched_onecycle',
                 'sched_dsd'], default='sched_agp',
        help='Schedule function for the sparsifier. Choice is between'
             ' one_shot, iterative, sched_agp (default), sched_onecycle,'
             ' and sched_dsd.'
             ' N.B.: one_shot and iterative are meant to be used with a start'
             ' epoch, as they are not gradual pruning schedules. More info at:'
             ' https://nathanhubens.github.io/fasterai/schedules.html')
    parser.add_argument(
        '--sparsification_start_epoch', '--sparsifier_start_epoch',
        '--spars_start_epoch', dest='spars_start_epoch', type=int, default=0,
        help='Training epoch at which the scheduled sparsification starts'
             ' (default 0).')
    parser.add_argument(
       '--sparsification_end_epoch', '--sparsifier_end_epoch',
       '--spars_end_epoch', dest='spars_end_epoch', type=int, default=None,
       help='Training epoch at which the scheduled sparsification ends.'
            ' If None (default), the sparsification ends at the end of the'
            ' training.')
    parser.add_argument(
       '--lottery-ticket-hypothesis', '--lottery-ticket', '--lth',
       dest='lth', action='store_true',
       help='Perform \'Lottery Ticket Hypothesis\'-style rewinding after every'
            ' pruning operation during the scheduled sparsification.'
            ' Original LTH paper (Frankle and Carbin, 2019):'
            ' https://arxiv.org/abs/1803.03635')
    parser.add_argument(
       '--no-lottery-ticket-hypothesis', '--no-lottery-ticket', '--no-lth',
       dest='lth', action='store_false',
       help='Do not perform \'Lottery Ticket Hypothesis\'-style rewinding'
            ' after every pruning operation during the scheduled'
            ' sparsification (default option).')
    parser.set_defaults(lth=False)
    parser.add_argument(
       '--lth_rewind_epoch', '--lth_rewind', '--rewind_epoch',
       dest='lth_rewind_epoch', type=int, default=0,
       help='Reference training epoch for LTH-style rewinding (default 0).')

    # Sparsification parameters that are specific to each schedule function.
    parser.add_argument(
        '--sparsification_iterative_n_steps', '--sparsifier_iterative_n_steps',
        '--spars_iterative_n_steps', '--iterative_n_steps',
        dest='spars_iterative_n_steps', type=int, default=3,
        help='Number of steps for the iterative schedule function'
             ' (default 3).')
    parser.add_argument(
        '--sparsification_sched_onecycle_alpha',
        '--sparsifier_sched_onecycle_alpha', '--spars_sched_onecycle_alpha',
        '--sched_onecycle_alpha',
        '--sparsification_onecycle_alpha', '--sparsifier_onecycle_alpha',
        '--spars_onecycle_alpha', '--onecycle_alpha',
        dest='spars_sched_onecycle_alpha', type=float, default=14,
        help='Alpha value for the sched_onecycle schedule function'
             ' (default 14).')
    parser.add_argument(
        '--sparsification_sched_onecycle_beta',
        '--sparsifier_sched_onecycle_beta', '--spars_sched_onecycle_beta',
        '--sched_onecycle_beta',
        '--sparsification_onecycle_beta', '--sparsifier_onecycle_beta',
        '--spars_onecycle_beta', '--onecycle_beta',
        dest='spars_sched_onecycle_beta', type=float, default=6,
        help='Beta value for the sched_onecycle schedule function'
             ' (default 6).')
    parser.add_argument(
        '--sparsification_sched_dsd_middle', '--sparsifier_sched_dsd_middle',
        '--spars_sched_dsd_middle', '--sched_dsd_middle',
        '--sparsification_dsd_middle', '--sparsifier_dsd_middle',
        '--spars_dsd_middle', '--dsd_middle',
        dest='spars_sched_dsd_middle', type=float, default=None,
        help='Spasity percentage at the middle of the pruning for the'
             ' sched_dsd schedule function. If None (default), the percentage'
             ' corresponds to halfway between the specified end sparsity and'
             ' 100% (full zero-out).')
    parser.add_argument(
        '--sparsification_sched_dsd_pattern', '--sparsifier_sched_dsd_pattern',
        '--spars_sched_dsd_pattern', '--sched_dsd_pattern',
        '--sparsification_dsd_pattern', '--sparsifier_dsd_pattern',
        '--spars_dsd_pattern', '--dsd_pattern',
        dest='spars_sched_dsd_pattern', type=str,
        choices=['square', 'lin', 'triangle', 'cos', 'sine', 'poly', 'agp',
                 'poly_wave', 'agp_wave', 'onecycle'],
        default='cos',
        help='Pattern of the pruning and unpruning motion for the sched_dsd'
             ' schedule function. The choice is between \'square\' i.e. a'
             ' square wave function, \'lin\' or \'triangle\' i.e. a linear or'
             ' triangle wave function, \'cos\' or \'sine\' (default) i.e. a'
             ' (co)sine wave function, functions based on the \'poly\' LR'
             ' schedule and the \'agp\' and \'onecycle\' pruning schedules,'
             ' and \'poly_wave\' and \'agp_wave\' adaptations as periodic'
             ' wave functions.')
    parser.add_argument(
        '--sparsification_sched_dsd_iterations',
        '--sparsifier_sched_dsd_iterations',
        '--spars_sched_dsd_iterations', '--sched_dsd_iterations',
        '--sparsification_dsd_iterations', '--sparsifier_dsd_iterations',
        '--spars_dsd_iterations', '--dsd_iterations',
        dest='spars_sched_dsd_iterations', type=int, default=1,
        help='Number of pruning and unpruning iterations for the sched_dsd'
             ' schedule function (default 1).')
    parser.add_argument(
        '--sparsification_sched_dsd_middle_pos',
        '--sparsifier_sched_dsd_middle_pos',
        '--spars_sched_dsd_middle_pos', '--sched_dsd_middle_pos',
        '--sparsification_dsd_middle_pos', '--sparsifier_dsd_middle_pos',
        '--spars_dsd_middle_pos', '--dsd_middle_pos',
        dest='spars_sched_dsd_middle_pos', type=float, default=0.5,
        help='Relative position (between 0 and 1) corresponding to \'the'
             ' middle of the pruning\' for the sched_dsd schedule function'
             ' (i.e. where sparsity = spars_sched_dsd_middle) (default 0.5).')

    # Learning rate reduction schedules (mostly used for prebuilt networks).
    parser.add_argument(
        '--reduce_learning_rate_start_epoch', '--rlr_start_epoch',
        dest='rlr_start_epoch', type=int, default=0,
        help='If the learning rate is modified, number of epochs elapsed'
             ' between the activation of scheduled LR modifications and the'
             ' actual start of the schedule.')
    parser.add_argument(
        '--reduce_learning_rate_end_epoch', '--rlr_end_epoch',
        dest='rlr_end_epoch', type=int, default=None,
        help='If the learning rate is modified, number of epochs elapsed'
             ' between the activation of scheduled LR modifications and the'
             ' end of the schedule. A None value (default) is interpreted as'
             ' the length of a \'patience\' cycle if using a DensEMANN'
             ' variant that features them, or otherwise the value of n_epochs')
    parser.add_argument(
        '--prebuilt_reduce_lr', '--prebuilt_rlr',
        '--no_self_constructing_reduce_lr', '--no_self_constructing_rlr',
        '--no_self_constr_reduce_lr', '--no_self_constr_rlr', '-prlr',
        dest='prebuilt_rlr', type=str,
        choices=['DensEMANN', 'lin', 'cos', 'exp', 'poly'],
        default='DensEMANN',
        help='Choice on the learning rate reduction schedule to apply when'
             ' training prebuilt networks. The choice is between \'DensEMANN\''
             ' (default, i.e. the schedule that is used for DensEMANN\'s'
             ' learning rate reduction), \'lin\' i.e. linear, \'cos\' i.e.'
             ' cosine, \'exp\' i.e. exponential, and \'poly\' i.e. polynomial'
             ' with degree 0.5.')

    # Other.
    parser.add_argument(
        '--DensEMANN-weight-init', '--DensEMANN-init',
        dest='DensEMANN_init', action='store_true',
        help='Use DensEMANN\'s weight initialisation method for'
             ' prebuilt DenseNets.')
    parser.add_argument(
        '--standard-weight-init', '--standard-init',
        dest='DensEMANN_init', action='store_false',
        help='Do not use DensEMANN\'s weight initialisation method for'
             ' prebuilt DenseNets (default option).')
    parser.set_defaults(DensEMANN_init=False)

    # LOGS AND SAVES RELATED PARAMETERS ---------------------------------------
    # -------------------------------------------------------------------------

    # Whether or not to save the model's state (to load it back in the future).
    parser.add_argument(
        '--model-saves', '--saves',
        dest='should_save_model', action='store_true',
        help='Save the model (and relevant hyperparameters) during training'
             ' (default option).')
    parser.add_argument(
        '--no-model-saves', '--no-saves',
        dest='should_save_model', action='store_false',
        help='Do not save the model or its hyperparameters during training.')
    parser.set_defaults(should_save_model=True)
    # Wether or not to write CSV feature logs.
    parser.add_argument(
        '--feature-logs', '--ft-logs', '--logs',
        dest='should_save_ft_logs', action='store_true',
        help='Record the evolution of feature values in a CSV feature log'
             ' (default option).')
    parser.add_argument(
        '--no-feature-logs', '--no-ft-logs', '--no-logs',
        dest='should_save_ft_logs', action='store_false',
        help='Do not record feature values in a CSV feature log.')
    parser.set_defaults(should_save_ft_logs=True)

    # Parameters related to model saves.
    parser.add_argument(
        '--save-model-every-epoch', '--save-every-epoch', '--every-epoch',
        dest='save_model_every_epoch', action='store_true',
        help='Save the model every epoch (or rather, every time a ft-log entry'
             ' is written).')
    parser.add_argument(
        '--save-model-every-improvement', '--save-every-improvement',
        '--every-improvement', '--save-model-every-impr', '--save-every-impr',
        '--every-impr',
        '--no-save-model-every-epoch', '--no-save-every-epoch',
        '--no-every-epoch',
        dest='save_model_every_epoch', action='store_false',
        help='Save the model only when the validation loss improves, or'
             ' whenever DensEMANN requires it (default option).'
             ' N.B.: This is only functional when a validation set is used'
             ' (i.e. when valid_size is not set to 0).')
    parser.set_defaults(save_model_every_epoch=False)
    parser.add_argument(
        '--save_model_every_epoch_until', '--save_every_epoch_until',
        '--every_epoch_until',
        dest='every_epoch_until', type=int, default=None,
        help='Used for setting a limited number of epochs during which the'
             ' model is saved every epoch. After this limit, the model is only'
             ' saved if there is an improvement in the validation loss, or'
             ' when DensEMANN requires it. A None value (default) means that'
             ' there is no such limit.'
             ' N.B.: If this parameter is set to an integer value,'
             ' save_model_every_epoch is automatically set to True.')
    parser.add_argument(
        '--keep-intermediary-model-saves', '--keep-intermediary-saves',
        '--intermediary-model-saves', '--intermediary-saves',
        dest='keep_intermediary_model_saves', action='store_true',
        help='Keep intermediary model saves after using DensEMANN'
             ' (i.e. before pruning, before the last layer addition).'
             ' Implies should_save_model is set to true (via --model-saves).')
    parser.add_argument(
        '--no-keep-intermediary-model-saves', '--no-keep-intermediary-saves',
        '--no-intermediary-model-saves', '--no-intermediary-saves',
        dest='keep_intermediary_model_saves', action='store_false',
        help='Do not keep intermediary model saves after using DensEMANN.'
             ' This means that, if should_save_model is set to true, such'
             ' intermediary saves are deleted after running the algorithm'
             ' (default option).')
    parser.set_defaults(keep_intermediary_model_saves=False)

    # Parameters related to feature logs.
    parser.add_argument(
        '--feature_frequency', '--feature_freq', '--ft_freq', '-ffq',
        '--feature_period', '--ft_period', '-fp',
        dest='ft_freq', type=int, default=1,
        help='Number of epochs between two entries in feature logs, containing'
             ' measurements of values (e.g. epoch, accuracy, loss, CS),'
             ' and between two model saves if save_model_every_epoch is True'
             ' (default 1).')
    parser.add_argument(
        '--ft_comma_separator', '--ft_comma', '-comma',
        dest='ft_comma', type=str, default=';',
        help='Comma (value) separator for the CSV feature log'
             ' (default \';\').')
    parser.add_argument(
        '--ft_decimal_separator', '--ft_decimal', '-dec',
        dest='ft_decimal', type=str, default=',',
        help='Decimal separator for the CSV feature log'
             ' (default \',\').')
    parser.add_argument(
        '--add-feature-kCS', '--add-ft-kCS', '--feature-kCS', '--ft-kCS',
        dest='add_ft_kCS', action='store_true',
        help='Save kCS values from filters in each layer (i.e. the CS for each'
             ' filter) to the CSV feature log (default option).')
    parser.add_argument(
        '--no-add-feature-kCS', '--no-add-ft-kCS',
        '--no-feature-kCS', '--no-ft-kCS',
        dest='add_ft_kCS', action='store_false',
        help='Do not save kCS values to the CSV feature log.')
    parser.set_defaults(add_ft_kCS=True)

    args = parser.parse_args()

    # Perform settings depending on the parsed arguments.
    if not args.keep_prob:
        if args.dataset[-1] != "+":
            args.keep_prob = 0.8
        else:
            args.keep_prob = 1.0
    if args.valid_size is None:
        if args.dataset in ['SVHN', 'SVHN+']:
            args.valid_size = 6000
        elif args.dataset in ['ImageNet', 'ImageNet+']:
            args.valid_size = 128117
        elif args.dataset in ['FMNIST', 'FMNIST+']:
            args.valid_size = 6000
        elif args.dataset in ['FER2013', 'FER2013+']:
            args.valid_size = 3230
        else:  # C10, C10+, C100 and C100+
            args.valid_size = 5000
    if not args.train_size:
        if args.dataset in ['SVHN', 'SVHN+']:
            args.train_size = 6000
        elif args.dataset in ['ImageNet', 'ImageNet+']:
            args.train_size = 1153050
        elif args.dataset in ['FMNIST', 'FMNIST+']:
            args.train_size = 54000 if args.valid_size else 60000
        elif args.dataset in ['FER2013', 'FER2013+']:
            args.train_size = 29068 if args.valid_size else 32298
        else:  # C10, C10+, C100 and C100+
            args.train_size = 45000 if args.valid_size else 50000
    if args.model_type == 'DenseNet':
        args.reduction = 1.0

    # Handle commands: what to do, and whether or not to use an existing model.
    if not args.train and not args.test:
        raise Exception(
            'Operation on network (--train and/or --test) not specified!\n'
            'You should train or test your network. Please check arguments.')
    if not args.train and not args.source_experiment_id:
        raise Exception(
            'Operation --test specified without training,'
            ' but source model not specified!\n'
            'Please provide an experiment ID for the source model'
            ' (--source_experiment_id or --source_id).')

    # Build the controller and model from specified keyword args.
    controller = DensEMANN_controller(**vars(args))
    # Train and test the model from the controller.
    controller.run()
    print('Done!')
