import numpy as np

from sampling.direct_correlation import c2_from_g


def test_ideal_gas_has_zero_c2():
    r = np.linspace(0.05, 8.0, 160)
    g = np.ones_like(r)
    _, c2r, _, c2k = c2_from_g(r, g, rho=0.02)
    assert np.max(np.abs(c2r)) < 1e-10
    assert np.max(np.abs(c2k)) < 1e-10


def test_c2_roundtrip_shape():
    r = np.linspace(0.05, 8.0, 160)
    g = 1.0 + 0.5 * np.exp(-((r - 2.8) / 0.4) ** 2)
    r2, c2r, k, c2k = c2_from_g(r, g, rho=0.02)
    assert r2.shape == c2r.shape == r.shape
    assert k.shape == c2k.shape
    assert np.isfinite(c2r).all()
    assert np.isfinite(c2k).all()


def _transform(r, f):
    dr = r[1] - r[0]
    L = 2.0 * (r[-1] + dr)
    k = np.arange(0.0, np.pi / dr + 1e-12, 2.0 * np.pi / L)
    kr = np.outer(k, r)
    sinc = np.ones_like(kr)
    np.divide(np.sin(kr), kr, out=sinc, where=kr != 0)
    return k, 4.0 * np.pi * dr * ((r ** 2 * f) @ sinc.T), sinc, 2.0 * np.pi / L


def _inverse(k, fk, r, sinc, dk):
    return (dk / (2.0 * np.pi ** 2)) * ((k ** 2 * fk) @ sinc)


def test_oz_recovers_smooth_analytic_c():
    rho, sig, c0 = 0.2, 0.5, -5.0
    r = np.arange(0.0, 15.0, 0.01)
    c = c0 * np.exp(-0.5 * (r / sig) ** 2)
    k, ck, sinc, dk = _transform(r, c)
    h = _inverse(k, ck / (1.0 - rho * ck), r, sinc, dk)
    _, c2, _, _ = c2_from_g(r, 1.0 + h, rho)
    assert np.max(np.abs(c2 - c)) < 1e-8


def test_oz_recovers_py_hard_sphere_smooth_part():
    eta = 0.4
    rho = 6.0 * eta / np.pi
    r = np.arange(0.0, 20.0, 0.01)
    x = r
    c = np.zeros_like(r)
    m = x < 1.0
    c[m] = -((1 + 2 * eta) ** 2 - 6 * eta * (1 + eta / 2) ** 2 * x[m]
             + 0.5 * eta * (1 + 2 * eta) ** 2 * x[m] ** 3) / (1 - eta) ** 4
    k, ck, sinc, dk = _transform(r, c)
    h = _inverse(k, ck / (1.0 - rho * ck), r, sinc, dk)
    _, c2, _, _ = c2_from_g(r, 1.0 + h, rho)
    mask = (r > 0.2) & (r < 0.9)
    assert np.max(np.abs(c2[mask] - c[mask])) < 1e-8
