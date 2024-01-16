import torch
import numpy as np
import scipy as sp
from tqdm import tqdm
from functools import partial
from termcolor import colored

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.test_utils import my_tests
from easyvolcap.utils.loss_utils import lossfun_outer, inner_outer, lossfun_distortion, interval_distortion
from easyvolcap.utils.prop_utils import importance_sampling, weighted_percentile, searchsorted, max_dilate_weights, max_dilate, query, anneal_weights


@catch_throw
def assert_true(expr):
    if isinstance(expr, torch.Tensor):
        expr = expr.all()
    assert expr, f'{repr(expr)} is not true'


@catch_throw
def assert_func(func, *args, **kwargs):
    return func(*args, **kwargs)


def inner(t0, t1, w1):
    """A reference implementation for computing the inner measure of (t1, w1)."""
    w0_inner = []
    for i in range(len(t0) - 1):
        w_sum = 0
        for j in range(len(t1) - 1):
            if (t1[j] >= t0[i]) and (t1[j + 1] < t0[i + 1]):
                w_sum += w1[j]
        w0_inner.append(w_sum)
    w0_inner = torch.tensor(w0_inner)
    return w0_inner


# translation function from pytorch grammar to numpy grammar
def torch_randint(shape, minval, maxval):
    return torch.randint(minval, maxval, shape)


def torch_uniform(shape, minval=0., maxval=1.):
    return torch.rand(shape) * (maxval - minval) + minval


def torch_normal(shape=()):
    return torch.normal(0., 1., size=shape)


def torch_sorted(v, axis=-1):
    return torch.sort(v, dim=axis)[0]


def torch_mean(v: torch.Tensor):
    if v.dtype == torch.bool:
        return torch.mean(v.float())
    return torch.mean(v)


def torch_cumsum(v: torch.Tensor, axis=-1):
    return torch.cumsum(v, dim=axis)


def torch_maximum(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    return torch.maximum(a, b)


def torch_mininum(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    return torch.minimum(a, b)


def torch_softmax(v, axis=-1):
    return torch.softmax(v, axis)


def outer(t0, t1, w1):
    """A reference implementation for computing the outer measure of (t1, w1)."""
    w0_outer = []
    for i in range(len(t0) - 1):
        w_sum = 0
        for j in range(len(t1) - 1):
            if (t1[j + 1] >= t0[i]) and (t1[j] <= t0[i + 1]):
                w_sum += w1[j]
        w0_outer.append(w_sum)
    w0_outer = torch.tensor(w0_outer)
    return w0_outer


def test_searchsorted_in_bounds():
    """Test that a[i] <= v < a[j], with (i, j) = searchsorted(a, v)."""
    eps = 1e-7
    for _ in range(10):
        n = torch.randint(10, 100, ())
        m = torch.randint(10, 100, ())
        v = torch.rand((n,)) * (1 - eps - eps) + eps
        a, _ = torch.sort(torch.rand((m,)))
        a = torch.cat([torch.tensor([0., ]), a, torch.tensor([1., ])])
        idx_lo, idx_hi = searchsorted(a, v)
        assert_true(torch.all(a[idx_lo] <= v))
        assert_true(torch.all(a[idx_hi] > v))


def test_searchsorted_out_of_bounds():
    """searchsorted should produce the first/last indices when out of bounds."""
    for _ in range(10):
        n = torch.randint(10, 100, ())
        m = torch.randint(10, 100, ())
        a, _ = torch.sort(torch.rand((m,)) + 1.0)
        v_lo = torch.rand((n,)) * 0.9
        v_hi = torch.rand((n, )) * (3 - 2.1) + 2.1
        idx_lo, idx_hi = searchsorted(a, v_lo)
        assert_true(torch.all(idx_lo == 0))
        assert_true(torch.all(idx_hi == 0))
        idx_lo, idx_hi = searchsorted(a, v_hi)
        assert_true(torch.all(idx_lo == m - 1))
        assert_true(torch.all(idx_hi == m - 1))


def test_searchsorted_reference():
    """Test against torch.searchsorted, which behaves similarly to ours."""
    eps = 1e-7
    n = 30
    m = 40

    # Generate query points in [eps, 1-eps].
    v = torch.rand([n]) * (1 - eps - eps) + eps

    # Generate sorted reference points that span all of [0, 1].
    a, _ = torch.sort(torch.rand([m]))
    a = torch.cat([torch.tensor([0.]), a, torch.tensor([1.])])
    _, idx_hi = searchsorted(a, v)
    assert_true((np.array_equal(np.searchsorted(a, v), idx_hi.numpy())))


def test_searchsorted():
    """An alternative correctness test for in-range queries to searchsorted."""
    a, _ = torch.sort(torch_uniform([10], minval=-4, maxval=4))

    v = torch_uniform([100], minval=-6, maxval=6)

    idx_lo, idx_hi = searchsorted(a, v)

    for x, i0, i1 in zip(v, idx_lo, idx_hi):
        if x < torch.min(a):
            i0_true, i1_true = [0] * 2
        elif x > torch.max(a):
            i0_true, i1_true = [len(a) - 1] * 2
        else:
            i0_true = torch.argmax(torch.where(x >= a, a, -torch.inf))
            i1_true = torch.argmin(torch.where(x < a, a, torch.inf))
        assert_func(np.testing.assert_array_equal, i0_true, i0)
        assert_func(np.testing.assert_array_equal, i1_true, i1)


def impl_test_lossfun_outer(num_ablate, is_all_zero):
    """Two histograms of the same/diff points have a loss of zero/non-zero."""
    eps = 1e-12  # Need a little slack because of cumsum's numerical weirdness.
    all_zero = True
    for _ in range(10):
        num_pts, d0, d1 = torch_randint([3], minval=10, maxval=20)
        t0 = torch_sorted(torch_uniform([d0 + 1]), axis=-1)
        t1 = torch_sorted(torch_uniform([d1 + 1]), axis=-1)
        lo = torch_maximum(torch.min(t0), torch.min(t1)) + 0.1
        hi = torch_mininum(torch.max(t0), torch.max(t1)) - 0.1
        rand = torch_uniform([num_pts], minval=lo, maxval=hi)
        pts = rand
        pts_ablate = rand[:-num_ablate] if num_ablate > 0 else pts
        w0 = []
        for i in range(len(t0) - 1):
            w0.append(torch_mean((pts_ablate >= t0[i]) & (pts_ablate < t0[i + 1])))
        w0 = torch.tensor(w0)
        w1 = []
        for i in range(len(t1) - 1):
            w1.append(torch_mean((pts >= t1[i]) & (pts < t1[i + 1])))
        w1 = torch.tensor(w1)
        all_zero &= torch.all(lossfun_outer(t0, w0, t1, w1) < eps)
    assert_true(is_all_zero == all_zero)


test_lossfun_outer_sameset = partial(impl_test_lossfun_outer, 0, True)
test_lossfun_outer_diffset = partial(impl_test_lossfun_outer, 2, False)


def test_inner_outer():
    """Two histograms of the same points will be bounds on each other."""
    for _ in range(10):
        d0, d1, num_pts = torch_randint([3], minval=10, maxval=20)
        t0 = torch_sorted(torch_uniform([d0 + 1]), axis=-1)
        t1 = torch_sorted(torch_uniform([d1 + 1]), axis=-1)
        lo = torch_maximum(torch.min(t0), torch.min(t1)) + 0.1
        hi = torch_mininum(torch.max(t0), torch.max(t1)) - 0.1
        pts = torch_uniform([num_pts], minval=lo, maxval=hi)
        w0 = []
        for i in range(len(t0) - 1):
            w0.append(torch.sum((pts >= t0[i]) & (pts < t0[i + 1])))
        w0 = torch.tensor(w0)
        w1 = []
        for i in range(len(t1) - 1):
            w1.append(torch.sum((pts >= t1[i]) & (pts < t1[i + 1])))
        w1 = torch.tensor(w1)
        w0_inner, w0_outer = inner_outer(t0, t1, w1)
        w1_inner, w1_outer = inner_outer(t1, t0, w0)
        assert_true(torch.all(w0_inner <= w0) and torch.all(w0 <= w0_outer))
        assert_true(torch.all(w1_inner <= w1) and torch.all(w1 <= w1_outer))


def test_lossfun_outer_monotonic():
    """The loss is invariant to monotonic transformations on `t`."""
    def curve_fn(x): return 1 + x**3  # Some monotonic transformation.

    for _ in range(10):
        d0, d1 = torch_randint([2], minval=10, maxval=20)
        t0 = torch_sorted(torch_uniform([d0 + 1]), axis=-1)
        t1 = torch_sorted(torch_uniform([d1 + 1]), axis=-1)
        w0 = torch.exp(torch_normal([d0]))
        w1 = torch.exp(torch_normal([d1]))

        excess = lossfun_outer(t0, w0, t1, w1)
        curve_excess = lossfun_outer(curve_fn(t0), w0, curve_fn(t1), w1)
        assert_true(torch.all(excess == curve_excess))


def test_lossfun_outer_self_zero():
    """The loss is ~zero for the same (t, w) step function."""
    for _ in range(10):
        d = torch_randint((), minval=10, maxval=20)
        t = torch_sorted(torch_uniform([d + 1]), axis=-1)
        w = torch.exp(torch_normal([d]))
        assert_true(torch.all(lossfun_outer(t, w, t, w) < 1e-10))


def test_outer_measure_reference():
    """Test that outer measures match a reference implementation."""
    for _ in range(10):
        d0, d1 = torch_randint([2], minval=10, maxval=20)
        t0 = torch_sorted(torch_uniform([d0 + 1]), axis=-1)
        t1 = torch_sorted(torch_uniform([d1 + 1]), axis=-1)
        w0 = torch.exp(torch_normal([d0]))
        _, w1_outer = inner_outer(t1, t0, w0)
        w1_outer_ref = outer(t1, t0, w0)
        assert_func(np.testing.assert_allclose, w1_outer, w1_outer_ref, atol=1E-5, rtol=1E-5)


def test_inner_measure_reference():
    """Test that inner measures match a reference implementation."""
    for _ in range(10):
        d0, d1 = torch_randint([2], minval=10, maxval=20)
        t0 = torch_sorted(torch_uniform([d0 + 1]), axis=-1)
        t1 = torch_sorted(torch_uniform([d1 + 1]), axis=-1)
        w0 = torch.exp(torch_normal([d0]))
        w1_inner, _ = inner_outer(t1, t0, w0)
        w1_inner_ref = inner(t1, t0, w0)
        assert_func(np.testing.assert_allclose, w1_inner, w1_inner_ref, rtol=1e-5, atol=1e-5)


def impl_test_sample_train_mode(randomized, single_jitter):
    """Test that piecewise-constant sampling reproduces its distribution."""
    batch_size = 4
    num_bins = 16
    num_samples = 1000000
    precision = 1e5

    # Generate a series of random PDFs to sample from.
    data = []
    for _ in range(batch_size):
        # Randomly initialize the distances between bins.
        # We're rolling our own fixed precision here to make cumsum exact.
        bins_delta = torch.round(precision * torch.exp(
            torch_uniform(shape=(num_bins + 1,), minval=-3, maxval=3)))

        # Set some of the bin distances to 0.
        bins_delta *= torch_uniform(shape=bins_delta.shape) < 0.9

        # Integrate the bins.
        bins = torch_cumsum(bins_delta) / precision
        bins += torch_normal() * num_bins / 2

        # Randomly generate weights, allowing some to be zero.
        weights = torch_maximum(
            0, torch_uniform(shape=(num_bins,), minval=-0.5, maxval=1.))
        gt_hist = weights / weights.sum()
        data.append((bins, weights, gt_hist))

    bins, weights, gt_hist = [torch.stack(x) for x in zip(*data)]

    # Draw samples from the batch of PDFs.
    samples = importance_sampling(
        bins,
        torch_softmax(weights.log() + 0.7),
        num_samples,
        perturb=randomized,
        single_jitter=single_jitter,
    )
    assert_true(samples.shape[-1] == num_samples)

    # Check that samples are sorted. (sometimes this won't pass...)
    assert_func(np.testing.assert_array_compare, lambda x, y: x >= y, samples[..., 1:], samples[..., :-1])
    # (?<=\s)(np\.testing\.\w*)\(
    # assert_func($1,

    # Verify that each set of samples resembles the target distribution.
    for i_samples, i_bins, i_gt_hist in zip(samples, bins, gt_hist):
        i_hist = torch.histogram(i_samples, i_bins)[0].float() / num_samples
        i_gt_hist = torch.tensor(i_gt_hist)

        # Merge any of the zero-span bins until there aren't any left.
        while torch.any(i_bins[:-1] == i_bins[1:]):
            # find first zero-span index
            j = int(torch.where(i_bins[:-1] == i_bins[1:])[0][0])

            # merge i_hist
            left = i_hist[:j]
            if j + 1 < len(i_hist):
                middle = torch.tensor([i_hist[j] + i_hist[j + 1]])
            else:
                middle = torch.empty((0,))
            if j + 2 < len(i_hist):
                right = i_hist[j + 2:]
            else:
                right = torch.empty((0,))
            i_hist = torch.cat([left, middle, right])

            # merge i_gt_hist
            left = i_gt_hist[:j]
            if j + 1 < len(i_gt_hist):
                middle = torch.tensor([i_gt_hist[j] + i_gt_hist[j + 1]])
            else:
                middle = torch.empty((0,))
            if j + 2 < len(i_gt_hist):
                right = i_gt_hist[j + 2:]
            else:
                right = torch.empty((0,))
            i_gt_hist = torch.cat([left, middle, right])

            # merge i_bins
            i_bins = torch.cat([i_bins[:j], i_bins[j + 1:]])

        # Angle between the two histograms in degrees.
        angle = 180 / torch.pi * torch.arccos(
            torch_mininum(
                1.,
                torch_mean((i_hist * i_gt_hist) /
                           torch.sqrt(torch_mean(i_hist**2) * torch_mean(i_gt_hist**2)))))
        # Jensen-Shannon divergence.
        m = (i_hist + i_gt_hist) / 2
        js_div = torch.sum(sp.special.kl_div(i_hist, m) + sp.special.kl_div(i_gt_hist, m)) / 2
        assert_true(angle <= 0.5)
        assert_true(js_div <= 1e-5)


test_sample_train_mode_deterministic = partial(impl_test_sample_train_mode, False, False)
test_sample_train_mode_random_single_jitter = partial(impl_test_sample_train_mode, True, True)
test_sample_train_mode_random_multiple_jitter = partial(impl_test_sample_train_mode, True, False)


def impl_test_sample_single_bin(randomized, single_jitter):
    """Test sampling when given a small `one hot' distribution."""
    num_samples = 625
    bins = torch.tensor([0, 1, 3, 6, 10], dtype=torch.float32)
    for i in range(len(bins) - 1):
        weights = torch.zeros(len(bins) - 1, dtype=torch.float32)
        weights[i] = 1.

        samples = importance_sampling(
            bins[None],
            weights[None],
            num_samples,
            perturb=randomized,
            single_jitter=single_jitter,
        )[0]

        # All samples should be within [bins[i], bins[i+1]].
        assert_true(torch.all(samples >= bins[i]))
        assert_true(torch.all(samples <= bins[i + 1]))


test_sample_single_bin_deterministic = partial(impl_test_sample_single_bin, False, False)
test_sample_single_bin_random_single_jitter = partial(impl_test_sample_single_bin, True, True)
test_sample_single_bin_random_multiple_jitter = partial(impl_test_sample_single_bin, True, False)


def impl_test_sample_sparse_delta(randomized, single_jitter):
    """Test sampling when given a large distribution with a big delta in it."""
    num_samples = 100
    num_bins = 100000
    bins = torch.arange(num_bins)
    weights = np.ones(len(bins) - 1)
    delta_idx = len(weights) // 2
    weights[delta_idx] = len(weights) - 1
    samples = importance_sampling(
        bins[None],
        torch_softmax(torch_maximum(1e-15, weights[None]).log()),
        num_samples,
        perturb=randomized,
        single_jitter=single_jitter,
    )[0]

    # All samples should be within the range of the bins.
    assert_true(torch.all(samples >= bins[0]))
    assert_true(torch.all(samples <= bins[-1]))

    # Samples modded by their bin index should resemble a uniform distribution.
    samples_mod = torch.fmod(samples, 1)
    assert_true(
        sp.stats.kstest(samples_mod, 'uniform', (0, 1)).statistic <= 0.2)

    # The delta function bin should contain ~half of the samples.
    in_delta = (samples >= bins[delta_idx]) & (samples <= bins[delta_idx + 1])
    assert_func(np.testing.assert_allclose, torch.mean(in_delta.float()), 0.5, atol=0.05)


test_sample_sparse_delta_deterministic = partial(impl_test_sample_sparse_delta, False, False)
test_sample_sparse_delta_random_single_jitter = partial(impl_test_sample_sparse_delta, True, True)
test_sample_sparse_delta_random_multiple_jitter = partial(impl_test_sample_sparse_delta, True, False)


def impl_test_sample_large_flat(randomized, single_jitter):
    """Test sampling when given a large flat distribution."""
    num_samples = 100
    num_bins = 100000
    bins = torch.arange(num_bins)
    weights = np.ones(len(bins) - 1)
    samples = importance_sampling(
        bins[None],
        torch_softmax(torch_maximum(1e-15, weights[None]).log()),
        num_samples,
        perturb=randomized,
        single_jitter=single_jitter,
    )[0]
    # All samples should be within the range of the bins.
    assert_true(torch.all(samples >= bins[0]))
    assert_true(torch.all(samples <= bins[-1]))

    # Samples modded by their bin index should resemble a uniform distribution.
    samples_mod = torch.fmod(samples, 1)
    assert_true(
        sp.stats.kstest(samples_mod, 'uniform', (0, 1)).statistic <= 0.2)

    # All samples should collectively resemble a uniform distribution.
    assert_true(
        sp.stats.kstest(samples, 'uniform', (bins[0], bins[-1])).statistic <= 0.2)


test_sample_large_flat_deterministic = partial(impl_test_sample_large_flat, False, False)
test_sample_large_flat_random_single_jitter = partial(impl_test_sample_large_flat, True, True)
test_sample_large_flat_random_multiple_jitter = partial(impl_test_sample_large_flat, True, False)


def test_distortion_loss_against_sampling():
    """Test that the distortion loss matches a stochastic approximation."""
    # Construct a random step function that defines a weight distribution.
    n, d = 10, 8
    t = torch_uniform(minval=-3, maxval=3, shape=(n, d + 1))
    t, _ = torch.sort(t, axis=-1)
    logits = 2 * torch_normal(shape=(n, d))

    # Compute the distortion loss.
    w = torch.softmax(logits, axis=-1)
    losses = lossfun_distortion(t, w)

    # Approximate the distortion loss using samples from the step function.
    samples = importance_sampling(t, torch_softmax(logits), 10000, single_jitter=False)
    losses_stoch = []
    for i in range(n):
        losses_stoch.append(torch_mean(torch.abs(samples[i][:, None] - samples[i][None, :])))
    losses_stoch = torch.tensor(losses_stoch)

    assert_func(np.testing.assert_allclose, losses, losses_stoch, atol=1e-4, rtol=1e-4)


def test_distortion_loss_against_interval_distortion():
    """Test that the distortion loss matches a brute-force alternative."""
    # Construct a random step function that defines a weight distribution.
    n, d = 3, 8
    t = torch_uniform(minval=-3, maxval=3, shape=(n, d + 1))
    t = torch_sorted(t, axis=-1)
    logits = 2 * torch_normal(shape=(n, d))

    # Compute the distortion loss.
    w = torch_softmax(logits, axis=-1)
    losses = lossfun_distortion(t, w)

    # Compute it again in a more brute-force way, but computing the weighted
    # distortion of all pairs of intervals.
    d = interval_distortion(t[..., :-1, None], t[..., 1:, None],
                            t[..., None, :-1], t[..., None, 1:])
    losses_alt = torch.sum(w[:, None, :] * w[:, :, None] * d, axis=[-1, -2])

    assert_func(np.testing.assert_allclose, losses, losses_alt, atol=1e-6, rtol=1e-4)


def test_interval_distortion_against_brute_force():
    n, d = 3, 7

    t0 = torch_uniform(minval=-3, maxval=3, shape=(n, d + 1))
    t0 = torch_sorted(t0, axis=-1)

    t1 = torch_uniform(minval=-3, maxval=3, shape=(n, d + 1))
    t1 = torch_sorted(t1, axis=-1)

    distortions = interval_distortion(t0[..., :-1], t0[..., 1:],
                                      t1[..., :-1], t1[..., 1:])

    distortions_brute = np.array(torch.zeros_like(distortions))
    for i in range(n):
        for j in range(d):
            distortions_brute[i, j] = torch.mean(
                torch.abs(
                    torch.linspace(t0[i, j], t0[i, j + 1], 5001)[:, None] -
                    torch.linspace(t1[i, j], t1[i, j + 1], 5001)[None, :]))
    assert_func(np.testing.assert_allclose,
                distortions, distortions_brute, atol=1e-6, rtol=1e-3)


def test_weighted_percentile():
    """Test that step function percentiles match the empirical percentile."""
    num_samples = 1000000
    for _ in range(10):
        d = torch_randint((), minval=10, maxval=20)
        ps = 100 * torch_uniform([3])
        t = torch.sort(torch_normal([d + 1]), dim=-1)[0]
        w = torch_softmax(torch_normal([d]))
        samples = importance_sampling(t, w, num_samples, single_jitter=False)
        true_percentiles = torch.from_numpy(np.percentile(samples, ps))
        our_percentiles = weighted_percentile(t, w, ps / 100)
        assert_func(np.testing.assert_allclose, our_percentiles, true_percentiles, rtol=1e-4, atol=1e-4)


def test_weighted_percentile_vectorized():
    shape = (3, 4)
    d = 128
    ps = 100 * torch_uniform((5,))
    t = torch_sorted(torch_normal(shape + (d + 1,)), axis=-1)
    w = torch_softmax(torch_normal(shape + (d,)))
    percentiles_vec = weighted_percentile(t, w, ps / 100)
    percentiles = []
    for i in range(shape[0]):
        percentiles.append([])
        for j in range(shape[1]):
            percentiles[i].append(weighted_percentile(t[i, j], w[i, j], ps / 100))
        percentiles[i] = torch.stack(percentiles[i])
    percentiles = torch.stack(percentiles)
    assert_func(np.testing.assert_allclose,
                percentiles_vec, percentiles, rtol=1e-5, atol=1e-5)


def test_max_dilate():
    """Compare max_dilate to a brute force test on queries of step functions."""
    n, d, dilation = 20, 8, 0.53

    # Construct a non-negative step function.
    t = torch_cumsum(
        torch_randint(minval=1, maxval=10, shape=(n, d + 1)),
        axis=-1) / 10
    w = torch_softmax(torch_normal(shape=(n, d)), axis=-1)

    # Dilate it.
    td, wd = max_dilate(t, w, dilation)

    # Construct queries at the midpoint of each interval.
    tq = (torch.arange((d + 4) * 10) - 2.5) / 10

    # Query the step function and its dilation.
    wq = query(tq[None], t, w)
    wdq = query(tq[None], td, wd)

    # The queries of the dilation must be the max of the non-dilated queries.
    mask = torch.abs(tq[None, :] - tq[:, None]) <= dilation
    for i in range(n):
        wdq_i = torch.max(mask * wq[i], axis=-1)[0]
        assert_func(np.testing.assert_array_equal, wdq[i], wdq_i)


def test_weight_annealing_zero_slope_noop():
    """Test that when annealing rate is 1.0, annealing is a noop"""
    n, d = 100, 500

    # Construct a non-negative step function.
    t = torch_cumsum(
        torch_randint(minval=1, maxval=10, shape=(n, d + 1)),
        axis=-1) / 10
    w = torch_softmax(torch_normal(shape=(n, d)), axis=-1)

    # Anneal the weight according to impl
    wn = anneal_weights(t, w, 1, 0)

    assert_func(np.testing.assert_allclose, w, wn, rtol=1e-5, atol=1e-5)


def impl_test_weight_annealing(train_frac, anneal_slope):
    """Test weight annealing function against a more brute force computation"""
    n, d = 100, 500

    # Construct a non-negative step function.
    t = torch_cumsum(
        torch_randint(minval=1, maxval=10, shape=(n, d + 1)),
        axis=-1) / 10
    w = torch_softmax(torch_normal(shape=(n, d)), axis=-1)

    # Anneal the weight according to impl
    wn = anneal_weights(t, w, train_frac, anneal_slope)

    def bias(x, s): return (s * x) / ((s - 1) * x + 1)
    anneal = bias(train_frac, anneal_slope)
    wt = w ** anneal / torch.sum(w ** anneal, dim=-1, keepdim=True) * torch.sum(w, dim=-1, keepdim=True)  # more brute force way

    assert_func(np.testing.assert_allclose, wn, wt, rtol=1e-5, atol=1e-5)


test_weight_annealing_high_high = partial(impl_test_weight_annealing, 0.9, 10)
test_weight_annealing_high_low = partial(impl_test_weight_annealing, 0.9, 0.1)
test_weight_annealing_low_high = partial(impl_test_weight_annealing, 0.1, 10)
test_weight_annealing_low_low = partial(impl_test_weight_annealing, 0.1, 0.1)


if __name__ == '__main__':
    my_tests(globals())
