# DensEMANN for Pytorch + Fastai
A PyTorch + Fastai implementation of the DensEMANN algorithm.

Based on :
- TensorFlow DensEMANN implementation by me: https://github.com/AntonioGarciaDiaz/Self-constructing_DenseNet
- PyTorch efficient DenseNet-BC implementation by Geoff Pleiss: https://github.com/gpleiss/efficient_densenet_pytorch
- TensorFlow DenseNet implementation by Illarion Khlestov: https://github.com/ikhlestov/vision_networks

DensEMANN is a growing algorithm for automatically building [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993>) (DenseNets).
Its inner working are inspired on [the EMANN self-structuring algorithm by Salomé and Bersini (1994).](https://ieeexplore.ieee.org/document/374473)
The most recent version of the algorithm is based on the evolution of the network's accuracy on a validation set, as well as on a feature of convolution filters called the "connection strength".

The **connection strength** (kCS) of a filter k is the mean of the absolute weights in k.
On basis of their kCS, filters can be called 'settled' (if their kCS has not changed much in awhile),
'useful' (if they are 'settled' and their kCS is above a certain 'usefulness threshold'), or
'useless' (if they are 'settled' and their kCS is below a certain 'uselessness threshold').

The most recent version of DensEMANN contains a macro-algorithm (that builds a one-block network layer by layer),
and a micro-algorithm (that builds the last layer by adding and/or pruning kernels from it):

- The **macro-algorithm** consists of a succession of three stages:

  - **First layer**: A new layer with ``growth_rate`` filters is added, and the micro-algorithm is run to build it (i.e. to add/prune kernels in it).

  - **Ascension**: A new layer with as many filters as the first one is added to the last block every ``ascension_threshold`` epochs.
    The ``ascension_threshold`` is a constant parameter with a default value of 10.
    The stage ends when the standard deviation of the last 50 measured accuracies falls below 0.1.

  - **Improvement**: The micro-algorithm is run on the last added layer, then the macro-algorithm checks
    if the accuracy has improved significantly since the previous layer addition.
    If so it adds a new layer (with either ``growth_rate`` filters or as many as the previous layer, respectively depending on whether ``--same-k`` or ``--update-k`` is used), else the algorithm ends.


- The **micro-algorithm** consists of a succession of three stages:

  - **Improvement**: countdown of m_patience_param epochs; if a new (settled) filter becomes useful
    (its kCS is above an automatically set usefulness threshold), add a filter and restart the countdown;
    if the countdown ends, wait until all filters have settled and end the stage.

  - **Pruning**: save the current accuracy and prune all useless filters
    (their kCS is below an automatically set uselessness threshold) to end the stage.

  - **Recovery**: if/when the learning reate is at its minimal value
    (optionally after resetting the learning rate to its initial value and reducing it), wait until reaching
    pre-pruning accuracy; then if there are any new useless filters wait for all filters to settle and
    return to pruning, else end the stage.

## Available versions

In addition to the latest version, previous versions (or variants) of the algorithm can be used by specifying parameters (see "Running the code" below).
The most recent variant is always the default version of the algorithm. The currently available variants are:

- **Variant 4 (DensEMANN v0.4):** the macro-algorithm does not feature an ascension stage, but the micro-algorithm has got one:
  in it, filters are added one by one every ``micro_ascension_threshold`` epochs until one of them settles. Also, having 'settled'
  is not a precondition for filters to be declared 'useful' or 'useless'.

- **Variant 5 (DensEMANN v1.0):** similar to variant 4, but the micro-algorithm no longer uses an ascension stage.

- **Variant 6 (DensEMANN v1.1):** similar to variant 5, but having 'settled' is a precondition for filters to be declared 'useful' or 'useless'.

- **Variant 7 (DensEMANN v1.2):** **This is the most recent version.** similar to variant 6, but the macro-algorithm now uses an ascension stage.


## Aside from DensEMANN

The program can also be used to import DenseNets (built by DensEMANN or otherwise) or to define their architectures manually.
These DenseNet can be further trained using a variety of learning rate (LR) reduction and sparsification methods.
Sparsification is possible thanks to [the Fasterai library by Nathan Hubens (2020).](https://github.com/nathanhubens/fasterai)


## Running the code

Two types of DenseNets are available:

- DenseNet (without bottleneck layers): ``-m 'DenseNet'``
- DenseNet-BC (with bottleneck layers): ``-m 'DenseNet-BC'``

The following datasets are available:

- CIFAR-10:  ``--dataset C10``
- CIFAR-100:  ``--dataset C100``
- SVHN:  ``--dataset SVHN``
- Fashion-MNIST:  ``--dataset FMNIST``
- FER-2013:  ``--dataset FER2013``

Adding a '+' sign after the dataset name (as in ``--dataset C10+``) adds data augmentation for the training and validation data
[as explained in the original DenseNet paper.](https://arxiv.org/abs/1608.06993)

**Example runs:**

``python run_DensEMANN.py --train --test -m DenseNet-BC -ds C10+ --DensEMANN -var 7 -rlr 0 -k 12``

Here the program uses DensEMANN (variant 7) to build a DenseNet-BC trained on the CIFAR-10 dataset with data augmentation.
The initial layer's growth rate is set to 12 filters. The resulting self-constructed network is then tested.

``python run_DensEMANN.py --train --test -m DenseNet-BC -ds SVHN --prebuilt -nep 600 -lnl 12,12,12 -k 12 -prlr DensEMANN``

Here the program trains a DenseNet-BC with 3 dense blocks and 12 layers in each (growth rate = 12) for 600 epochs on the SVHN dataset.
The LR reduction schedule is similar to the one used in DensEMANN: divide the initial LR (default 0.1) after 50% and 75% of the epochs.
The network retains the weights for which it obtained the best validation set loss, and it is tested with these weights.

``python run_DensEMANN.py --train --test -m DenseNet-BC -ds SVHN --prebuilt -nep 900 -lnl 12,12,12 -k 12 -prlr DensEMANN
--sparsify  --granularity filter --spars_sched_func sched_dsd --spars_end_epoch 300 --rlr_start_epoch 300 --every_epoch_until 300
--end_sparsity 70``

Here the program trains the same DenseNet-BC from above, but it also adds some sparsification.
The sparsification takes place during the first 300 epochs: it is at the granularity of filters, with a target percentage of 70%,
and it follows a custom schedule based on the principles of [Dense-Sparse-Dense training (Han et al, 2016)](https://arxiv.org/pdf/1607.04381.pdf)
The last 600 epochs are for normal training. The same LR reduction schedule from above is used.
The network retains the weight for which it obtained the best loss *after the 300 sparsification epochs*, and it is tested with these weights.
