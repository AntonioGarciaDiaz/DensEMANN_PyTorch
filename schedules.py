# Collection of custom pruning schedules for Fasterai.

from math import floor
from fastai.vision.all import sched_lin, sched_cos
from fasterai.sparse.all import sched_onecycle


def sched_square(start, end, pos):
    """
    Pruning schedule inspired on a square wave pattern. Prunes to end % of the
    features when pos = 0.5.
    If used as a DSD pattern, it is rather faithful to the original DSD paper.

    Args:
        start (int or float) - percentage corresponding to the network's
            initial sparsity.
        end (int or float) - percentage corresponding to the network's
            final sparsity.
        pos (float) - relative position within the pruning schedule.
    """
    return start if pos < 0.5 else end


def sched_poly(start, end, pos, power=0.5, start_pos=0.25, end_pos=1):
    """
    Pruning schedule inspired on the polynomial schedule from Fastai:
    https://github.com/fastai/fastai/blob/
    26f12325951eca7c3a228a2c296d6f46a4d8debd/fastai/callback/schedule.py#L64

    Args:
        start (int or float) - percentage corresponding to the network's
            initial sparsity.
        end (int or float) - percentage corresponding to the network's
            final sparsity.
        pos (float) - relative position within the pruning schedule.
        power (float) - power for the polinomial function (default 0.5).
        start_pos (float) - relative position at which the actual schedule
            starts (before that, the value is 'start') (default 0.25).
        end_pos (float) - relative position at which the actual schedule
            ends (before that, the value is 'end') (default 1).
    """
    if pos < start_pos:
        return start
    elif pos > end_pos:
        return end
    else:
        true_pos = (pos - start_pos) / (end_pos - start_pos)
        return start + (end - start) * true_pos ** power


def sched_poly_wave(start, end, pos, power=0.5):
    """
    Pruning schedule inspired on the polynomial schedule from Fastai, but
    adapted into a periodic wave function.

    Args:
        start (int or float) - percentage corresponding to the network's
            initial sparsity.
        end (int or float) - percentage corresponding to the network's
            final sparsity.
        pos (float) - relative position within the pruning schedule.
        power (float) - power for the polinomial function (default 0.5).
    """
    middle = start + (end-start)/2
    if pos < 0.5:
        return middle - (start + (middle - start) * (2*(0.5-pos)) ** power)
    else:
        return middle + (end - middle) * (2*(pos-0.5)) ** power


def sched_agp(start, end, pos, power=3, start_pos=0.25, end_pos=1):
    """
    A generalisation of the Automated Gradual Pruning (AGP) schedule
    (Zhu and Gupta, 2017): https://arxiv.org/abs/1710.01878v2

    Args:
        start (int or float) - percentage corresponding to the network's
            initial sparsity.
        end (int or float) - percentage corresponding to the network's
            final sparsity.
        pos (float) - relative position within the pruning schedule.
        power (float) - power for the AGP function (default 3).
        start_pos (float) - relative position at which the actual schedule
            starts (before that, the value is 'start') (default 0.25).
        end_pos (float) - relative position at which the actual schedule
            ends (before that, the value is 'end') (default 1).
    """
    if pos < start_pos:
        return start
    elif pos > end_pos:
        return end
    else:
        true_pos = (pos - start_pos) / (end_pos - start_pos)
        return end + (start - end) * (1 - pos) ** power


def sched_agp_wave(start, end, pos, power=3):
    """
    Pruning schedule inspired on a generalisation of the AGP schedule, but
    adapted into a periodic wave function.

    Args:
        start (int or float) - percentage corresponding to the network's
            initial sparsity.
        end (int or float) - percentage corresponding to the network's
            final sparsity.
        pos (float) - relative position within the pruning schedule.
        power (float) - power for the AGP function (default 3).
    """
    middle = start + (end-start)/2
    if pos < 0.5:
        return middle - (middle + (start-middle) * (1 - 2*(0.5-pos)) ** power)
    else:
        return end + (middle-end) * (1 - 2*(pos-0.5)) ** power


def sched_dsd(start, end, pos, middle=None, pattern='cos', iterations=1,
              middle_pos=0.5, **kwargs):
    """
    Pruning schedule inspired on Dense-Sparse-Dense: prunes to middle % of the
    features, then unprunes to end %, following a certain pattern.
    DSD original paper (Han et al, 2016): https://arxiv.org/pdf/1607.04381.pdf
    Implementation based on the Fasterai tutorial at:
    https://nathanhubens.github.io/fasterai/schedules.html#Dense-Sparse-Dense

    Args:
        start (int or float) - percentage corresponding to the network's
            initial sparsity.
        end (int or float) - percentage corresponding to the network's
            final sparsity.
        pos (float) - relative position within the pruning schedule.
        middle (int or float or None) - percentage corresponding to the
            network's sparsity at the middle of the pruning (should be lower
            than both start and end), if None (default) corresponds to halfway
            between end and 100%.
        pattern (str) - the pattern that the pruning and unpruning motion
            follows, a choice between 'lin', 'cos', 'square', 'poly', 'agp',
            and 'onecycle' (default 'cos').
            Each pattern corresponds to a schedule function (either custom or
            from Fastai or Fasterai) whose name is 'sched_' + the pattern name.
        iterations (str) - number of DSD iterations (i.e., the times that the
            pattern is repeated) (default 1).
        middle_pos (float) - relative position corresponding to the 'middle'
            sparsity (default 0.5).
        **kwargs (dict) - contains pattern-specific keyword arguments (which
            are passed to the corresponding 'sched_' function).

    Raises:
        Exception: pattern 'pattern' not available.
            Available patterns: ['square', 'lin', 'cos', 'poly', 'agp',
            'poly_wave', 'agp_wave', 'onecycle']
    """
    # Manage alternative names for some patterns.
    alt_pattern_names = {'triangle': 'lin', 'sine': 'cos'}
    if pattern in alt_pattern_names:
        pattern = alt_pattern_names[pattern]
    # Check among the available patterns.
    available_patterns = ['square', 'lin', 'cos', 'poly', 'agp', 'poly_wave',
                          'agp_wave', 'onecycle']
    if pattern not in available_patterns:
        raise Exception(('pattern \'{}\' not available. ' +
                        'Available patterns: {}.').format(
                            pattern, str(available_patterns)))
    if middle is None:
        middle = end + (100-end)/2
    # Set true_pos on basis of iterations.
    true_pos = (pos * iterations) % 1
    if true_pos < middle_pos:
        # We follow the pattern forward.
        return eval('sched_%s(start, middle, true_pos/middle_pos,'
                    ' **kwargs)' % pattern)
    else:
        # We follow the pattern backward.
        return eval('sched_%s(end if floor(pos * iterations) == iterations-1'
                    ' else start, middle, (1-true_pos)/(1-middle_pos),'
                    ' **kwargs)' % pattern)


def sched_dsd_original(start, end, pos, middle=None):
    """
    Pruning schedule inspired on Dense-Sparse-Dense: one-shot prunes middle %
    of the features, then one-shot unprunes to reach end %.
    This implementation of DSD-like pruning is more faithful to the original
    DSD paper (Han et al, 2016): https://arxiv.org/pdf/1607.04381.pdf

    Args:
        start (int or float) - percentage corresponding to the network's
            initial sparsity.
        end (int or float) - percentage corresponding to the network's
            final sparsity.
        pos (float) - relative position within the pruning schedule.
        middle (int or float or None) - percentage corresponding to the
            network's sparsity at the middle of the pruning (should be lower
            than both start and end), if None (default) corresponds to halfway
            between end and 100%.
    """
    if middle is None:
        middle = end + (100-end)/2
    if pos < 1:
        return middle
    else:
        return end
