#!/usr/bin/env python3
"""
Bayesian Modeling: PreliZ, PyMC, nutpie, NumPyro, ArviZ, and Bambi
==================================================================
Chooses a prior with PreliZ, then estimates a coin's bias from 50 flips with
PyMC and checks the posterior against the exact (conjugate) answer, sampling
it three ways: PyMC's own NUTS, nutpie (a Rust sampler), and NumPyro (JAX).
A hierarchical model of the classic "eight schools" data follows, summarized
and plotted with ArviZ, and Bambi fits a regression from an R-style formula.

PyMC compiles models with Numba (PyTensor's default backend), so no C compiler
is needed in the image.

PreliZ:  https://preliz.readthedocs.io/
PyMC:    https://www.pymc.io/
nutpie:  https://pymc-devs.github.io/nutpie/
NumPyro: https://num.pyro.ai/
ArviZ:   https://python.arviz.org/
Bambi:   https://bambinos.github.io/bambi/
"""

import json
import os
import time

import arviz as az
import bambi as bmb
import matplotlib
import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FLIPS, HEADS = 50, 36
SAMPLERS = ['pymc', 'nutpie', 'numpyro']
# cores=1 keeps PyMC's own sampler in this process. Run as a script, worker
# processes would re-execute this file: Python 3.14 starts them with forkserver
# on Linux. nutpie and NumPyro run their chains in-process either way.
SAMPLE_ARGS = {'draws': 1_000, 'tune': 1_000, 'chains': 2, 'cores': 1, 'progressbar': False, 'random_seed': 1}
summary: dict[str, object] = {}

# =============================================================================
# PreliZ: turn a belief into a prior
# =============================================================================
print("=" * 60)
print("PreliZ: Eliciting a Prior")
print("=" * 60)

# "I'm 90% sure the coin's heads probability lies between 0.3 and 0.7."
prior = pz.maxent(pz.Beta(), lower=0.3, upper=0.7, mass=0.9, plot=False)
alpha, beta = float(prior.alpha), float(prior.beta)
print(f"Maximum-entropy Beta with 90% of its mass in [0.3, 0.7]: Beta({alpha:.2f}, {beta:.2f})")

# =============================================================================
# The coin: three samplers against the exact posterior
# =============================================================================
print("\n" + "=" * 60)
print(f"Coin Flips: {HEADS} heads in {FLIPS}")
print("=" * 60)

exact = stats.beta(alpha + HEADS, beta + FLIPS - HEADS)  # Beta prior + binomial data stays Beta
print(f"Exact posterior: mean {exact.mean():.4f}, 89% interval {exact.ppf(0.055):.3f} to {exact.ppf(0.945):.3f}")

with pm.Model() as coin:
    p = pm.Beta('p', alpha=alpha, beta=beta)
    pm.Binomial('heads', n=FLIPS, p=p, observed=HEADS)

draws = {}
for sampler in SAMPLERS:
    started = time.perf_counter()
    with coin:
        idata = pm.sample(nuts_sampler=sampler, **SAMPLE_ARGS)
    seconds = time.perf_counter() - started
    draws[sampler] = idata.posterior['p'].values.ravel()
    print(f"  {sampler:8} mean {draws[sampler].mean():.4f}  "
          f"89% interval {np.quantile(draws[sampler], 0.055):.3f} to {np.quantile(draws[sampler], 0.945):.3f}  "
          f"({seconds:.1f} s, first call includes compilation)")
    if abs(draws[sampler].mean() - exact.mean()) > 0.01:
        raise SystemExit(f"{sampler} posterior mean is far from the exact answer")
summary['coin_posterior_mean'] = {name: round(float(values.mean()), 4) for name, values in draws.items()}

grid = np.linspace(0.3, 0.95, 300)
fig, ax = plt.subplots(figsize=(8, 4))
for sampler, values in draws.items():
    ax.hist(values, bins=40, density=True, histtype='step', lw=1.5, label=f'{sampler} draws')
ax.plot(grid, exact.pdf(grid), 'k--', label='exact posterior')
ax.plot(grid, stats.beta(alpha, beta).pdf(grid), color='gray', alpha=0.6, label='prior')
ax.set(xlabel='probability of heads', ylabel='density', title=f'{HEADS} heads in {FLIPS} flips')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'bayes_coin.png'), dpi=120)
plt.close()
print("Saved: bayes_coin.png")

# =============================================================================
# Hierarchical model: eight schools
# =============================================================================
print("\n" + "=" * 60)
print("PyMC + ArviZ: Eight Schools (Hierarchical)")
print("=" * 60)

schools = list('ABCDEFGH')
effects = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])  # estimated treatment effects
std_errors = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

with pm.Model(coords={'school': schools}) as eight_schools:
    mu = pm.Normal('mu', mu=0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    # Non-centered form: sample standard normals and scale them, which avoids the
    # funnel-shaped posterior that makes NUTS diverge in the centered version.
    offset = pm.Normal('offset', mu=0, sigma=1, dims='school')
    theta = pm.Deterministic('theta', mu + tau * offset, dims='school')
    pm.Normal('observed', mu=theta, sigma=std_errors, observed=effects, dims='school')
    # A higher target acceptance takes smaller steps through the narrow region near tau = 0.
    schools_idata = pm.sample(nuts_sampler='nutpie', target_accept=0.95, **SAMPLE_ARGS)

divergences = int(schools_idata.sample_stats['diverging'].sum())
table = az.summary(schools_idata, var_names=['mu', 'tau', 'theta'])
print(table)
print(f"Divergent transitions: {divergences}")
summary['eight_schools_divergences'] = divergences

forest = az.plot_forest(schools_idata, var_names=['theta'])
forest.savefig(os.path.join(OUTPUT_DIR, 'bayes_eight_schools.png'))
plt.close('all')
print("Saved: bayes_eight_schools.png (each school's effect, pulled toward the shared mean)")

# =============================================================================
# Bambi: a regression from a formula
# =============================================================================
print("\n" + "=" * 60)
print("Bambi: y ~ x + group")
print("=" * 60)

rng = np.random.default_rng(seed=0)
data = pd.DataFrame({'x': rng.normal(size=300), 'group': rng.choice(['control', 'treated'], size=300)})
data['y'] = 1.0 + 2.0 * data['x'] + 0.5 * (data['group'] == 'treated') + rng.normal(scale=0.5, size=300)

regression = bmb.Model('y ~ x + group', data)
fit = regression.fit(inference_method='nutpie', **SAMPLE_ARGS)
coefficients = az.summary(fit, var_names=['Intercept', 'x', 'group'])
print(coefficients)
print("True values: Intercept 1.0, x 2.0, group[treated] 0.5")
summary['bambi_means'] = {name: float(value) for name, value in coefficients['mean'].items()}

with open(os.path.join(OUTPUT_DIR, 'bayes_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved: bayes_summary.json")
print("Done.")
