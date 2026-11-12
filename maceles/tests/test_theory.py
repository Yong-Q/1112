"""Analytic checks of the FEP/importance-sampling estimator."""
import numpy as np

from sampling.off_polar import ReweightConfig

KB = 8.617333262e-5
T = 300.0
BETA = 1.0 / (KB * T)
K = 5.5e-18 / 1.602176634e-19  # eV/A^2
X0 = 0.02


def weights(dU):
    b = BETA * dU
    w = np.exp(-(b - b.min()))
    return w / w.sum()


def test_gaussian_fep_converges_to_exact_moments():
    rng = np.random.default_rng(0)
    sigma = 1.0 / np.sqrt(BETA * K)
    x = rng.normal(0.0, sigma, 200_000)
    dU = 0.5 * K * ((x - X0) ** 2 - x ** 2)
    w = weights(dU)
    mean = float(np.sum(w * x))
    x2 = float(np.sum(w * x ** 2))
    assert abs(mean - X0) < 2e-4
    assert abs(x2 - (X0 ** 2 + sigma ** 2)) < 2e-5


def test_constant_offset_is_invariant():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 5000)
    dU = 0.1 * x
    w1 = weights(dU)
    w2 = weights(dU + 1234.5)
    assert np.allclose(np.sort(w1), np.sort(w2), atol=1e-14)


def test_sigma_scales_sqrt_n():
    rng = np.random.default_rng(2)
    sigma = 1.0 / np.sqrt(BETA * K)
    masses = []
    for n in (1, 16, 256):
        x = rng.normal(0.0, sigma, (4000, n))
        dU = np.sum(0.5 * K * ((x - X0) ** 2 - x ** 2), axis=1)
        measured = float((BETA * dU).std(ddof=1))
        theory = X0 * np.sqrt(BETA * K * n)
        masses.append((measured, theory))
        assert abs(measured / theory - 1.0) < 0.05
    assert masses[1][0] > masses[0][0]
    assert masses[2][0] > masses[1][0]


def test_beta_config():
    cfg = ReweightConfig(temperature_K=300.0)
    assert abs(cfg.beta_eV_inv - BETA) < 1e-6
