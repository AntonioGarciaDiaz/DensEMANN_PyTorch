# Modifications of fastai callbacks for DensEMANN controller,
#
# Copyright notice:
# --------------------------------------------------------------------------
# Copyright 2019-2021 The Fastai Authors.
# Copyright 2021-2022 Antonio García-Díaz and Hugues Bersini.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------

import os
import torch
from fastai.vision.all import *


class CustomCSVLogger(Callback):
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
        ft_freq (int) - number of epochs between two entries in CSV logs
            (default 1).
        ft_comma (str) - 'comma' separator in the CSV logs (default ';').
        ft_decimal (str) - 'decimal' separator in the CSV logs (default ',').

    Attributes:
        append (bool) - whether or not to append lines to an existing CSV log
            (if the file exists, lines are appended to it).
        fname (str) - from args.
        add_ft_kCS (bool) - from args.
        ft_freq (int) - from args.
        ft_comma (str) - from args.
        ft_decimal (str) - from args.
    """
    order = 61

    def __init__(self, fname='history.csv', add_ft_kCS=True, ft_freq=1,
                 ft_comma=';', ft_decimal=','):
        """
        Initializer for the CustomCSVLogger callback.
        """
        self.fname = Path(fname)
        self.append = os.path.isfile(self.fname)
        self.add_ft_kCS = add_ft_kCS
        self.ft_freq = ft_freq
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
        During training, lines are only written every ft_freq epochs.
        """
        if not self.learn.training or self.learn.epoch % self.ft_freq == 0:
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


class CustomSaveModelCallback(SaveModelCallback):
    """
    Child class for the fastai SaveModelCallback callback.
    (https://github.com/fastai/fastai/blob/
    c47022bf0f8e106868d8b422801ecf18e9903d89/
    fastai/callback/tracker.py#L63)
    The main difference is that, when the best model is loaded back, key
    matches may not be strictly enforced.
    There are also differences concerning the behaviour in 'every_epoch'
    mode: the same model file is used for all epochs, and a new arg may be
    passed to specify a limited number of epochs during which 'every_epoch'
    has got an effect.

    Args:
        every_epoch_until (int or None) - optional number of epochs during
            which 'every_epoch' is considered True
            (default None, i.e. all epochs).
        strict (bool) - whether or not to strictly enforce key matches on
            loadback (default False).
        All args from SaveModelCallback.

    Attributes:
        every_epoch_until (int or None) - from args.
        strict (bool) - from args.
        All attributes from SaveModelCallback.
    """
    def __init__(self, monitor='valid_loss', comp=None, min_delta=0.,
                 fname='model', every_epoch=False, every_epoch_until=None,
                 at_end=False, with_opt=False, reset_on_fit=True,
                 strict=False):
        """
        Initializer for the CustomSaveModelCallback.
        """
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta,
                         fname=fname, every_epoch=every_epoch, at_end=at_end,
                         with_opt=with_opt, reset_on_fit=reset_on_fit)
        self.every_epoch_until = every_epoch_until
        self.strict = strict

    def after_epoch(self):
        """
        Different behaviour for every_epoch: always save the model to the same
        file instead of creating a new file at each epoch.
        """
        if self.every_epoch and (self.every_epoch_until is not None and
                                 self.epoch < self.every_epoch_until):
            if (self.epoch % self.every_epoch) == 0:
                self._save(f'{self.fname}')
        else:
            if self.every_epoch:
                self.every_epoch = False
            super().after_epoch()

    def after_fit(self, **kwargs):
        """
        Load the best model (unless at_end or every_epoch).
        """
        if self.at_end:
            self._save(f'{self.fname}')
        elif not self.every_epoch:
            self.learn.load(
                f'{self.fname}', with_opt=self.with_opt, strict=self.strict)
