#!/usr/bin/env python3
# # Bayesian A/B Test: Is the New Checkout Page Better?
#
# Two versions of a page, each shown to a few thousand visitors. Instead of a
# p-value, a Bayesian analysis answers the questions people actually ask: how
# likely is it that B beats A, by how much, and what do we lose if we pick B and
# it turns out worse? The same model is fitted twice: exactly (Beta-Binomial
# conjugacy, no sampling) and with PyMC, and the answers must agree.
#
# Replace `VISITORS` and `CONVERSIONS` with your own counts.
#
# - [PyMC](https://www.pymc.io/)
# - [ArviZ](https://python.arviz.org/)

import json
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy import stats

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

VISITORS = {'A': 4_820, 'B': 4_903}
CONVERSIONS = {'A': 512, 'B': 587}
PRIOR = (1, 1)  # Beta(1, 1): every conversion rate equally likely before seeing data
DRAWS = 200_000
SEED = 0
rng = np.random.default_rng(SEED)

for arm in VISITORS:
    print(f"{arm}: {CONVERSIONS[arm]} of {VISITORS[arm]} converted ({CONVERSIONS[arm] / VISITORS[arm]:.2%})")

# ## The exact posterior
#
# With a Beta prior and binomial data, the posterior of each conversion rate is
# again a Beta distribution, so we can draw from it directly.

posterior = {
    arm: stats.beta(PRIOR[0] + CONVERSIONS[arm], PRIOR[1] + VISITORS[arm] - CONVERSIONS[arm]) for arm in VISITORS
}
rate = {arm: dist.rvs(size=DRAWS, random_state=rng) for arm, dist in posterior.items()}
lift = rate['B'] / rate['A'] - 1

prob_b_better = float((rate['B'] > rate['A']).mean())
lift_low, lift_high = np.quantile(lift, [0.03, 0.97])
# Expected loss: the conversion rate we give up, on average, if we pick an arm and it is the worse one.
loss_if_b = float(np.maximum(rate['A'] - rate['B'], 0).mean())
loss_if_a = float(np.maximum(rate['B'] - rate['A'], 0).mean())

print(f"P(B beats A)          {prob_b_better:.1%}")
print(f"Relative lift of B    {np.median(lift):+.1%} (94% interval {lift_low:+.1%} to {lift_high:+.1%})")
print(f"Expected loss: choose B {loss_if_b:.5f}, choose A {loss_if_a:.5f} (conversion rate points)")

# ## A decision rule
#
# A common rule: ship B when the expected loss of choosing it is below a
# threshold you can live with, here 0.1 percentage points of conversion.

THRESHOLD = 0.001
decision = 'ship B' if loss_if_b < THRESHOLD else ('keep A' if loss_if_a < THRESHOLD else 'keep testing')
print(f"Decision with a {THRESHOLD:.1%} loss threshold: {decision}")

# ## The same model in PyMC
#
# The conjugate shortcut only works for this simple model. PyMC handles any
# model; here it must reproduce the exact answer, which is a useful check before
# extending the model (for example with a prior from past experiments).

with pm.Model(coords={'arm': list(VISITORS)}) as model:
    p = pm.Beta('p', alpha=PRIOR[0], beta=PRIOR[1], dims='arm')
    pm.Binomial('conversions', n=list(VISITORS.values()), p=p, observed=list(CONVERSIONS.values()), dims='arm')
    pm.Deterministic('lift', p[1] / p[0] - 1)
    idata = pm.sample(draws=2_000, tune=1_000, chains=2, cores=1, nuts_sampler='nutpie', progressbar=False,
                      random_seed=SEED)

pymc_prob = float((idata.posterior['p'].sel(arm='B') > idata.posterior['p'].sel(arm='A')).mean())
print(az.summary(idata, var_names=['p', 'lift']))
print(f"P(B beats A): exact {prob_b_better:.3f}, PyMC {pymc_prob:.3f}")
if abs(pymc_prob - prob_b_better) > 0.02:
    raise SystemExit("PyMC disagrees with the exact posterior")

# ## Plot the posteriors

grid = np.linspace(0.09, 0.135, 400)
fig, (ax_rate, ax_lift) = plt.subplots(1, 2, figsize=(11, 4))
for arm, dist in posterior.items():
    ax_rate.plot(grid, dist.pdf(grid), label=f'{arm}: {CONVERSIONS[arm] / VISITORS[arm]:.2%} observed')
ax_rate.set(xlabel='conversion rate', ylabel='density', title='Posterior conversion rates')
ax_rate.legend()
ax_lift.hist(lift, bins=120, density=True, color='#4c72b0', alpha=0.8)
ax_lift.axvline(0, color='black', lw=1)
ax_lift.set(xlabel='relative lift of B over A', title=f'P(B beats A) = {prob_b_better:.1%}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ab_posteriors.png'), dpi=120)
plt.show()

with open(os.path.join(OUTPUT_DIR, 'ab_summary.json'), 'w') as f:
    json.dump({'prob_b_better': prob_b_better, 'median_lift': float(np.median(lift)), 'loss_if_b': loss_if_b,
               'loss_if_a': loss_if_a, 'decision': decision}, f, indent=2)
print("Saved: ab_posteriors.png, ab_summary.json")
