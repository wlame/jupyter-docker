#!/usr/bin/env python3
# # Shift Scheduling with CP-SAT
#
# A small clinic needs a two-week rota: every shift needs enough staff, nobody
# works two shifts a day or a night followed by a morning, everyone gets a fair
# number of shifts, and requested days off are honoured where possible. OR-Tools'
# CP-SAT solver handles hard rules (must hold) and soft wishes (cost a penalty
# when broken) in one model, and proves when a schedule is optimal.
#
# Edit `STAFF`, `DEMAND`, and `DAYS_OFF` to model your own team.
#
# - [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver)

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAFF = ['Ana', 'Ben', 'Chen', 'Dara', 'Eli', 'Femi', 'Gus', 'Hana']
SHIFTS = ['morning', 'evening', 'night']
DAYS = 14
DEMAND = {'morning': 3, 'evening': 2, 'night': 1}  # staff needed per shift, every day
DAYS_OFF = {'Ana': [2, 3], 'Chen': [7], 'Eli': [0, 1, 13], 'Hana': [9, 10]}  # requests, soft
MIN_SHIFTS, MAX_SHIFTS = 9, 11  # per person over the two weeks
TIME_LIMIT_SECONDS = 20

# ## Decision variables
#
# `works[p, d, s]` is 1 when person p works shift s on day d.

model = cp_model.CpModel()
works = {(p, d, s): model.new_bool_var(f'{p}_{d}_{s}') for p in STAFF for d in range(DAYS) for s in SHIFTS}

# ## Hard rules

for d in range(DAYS):
    for s in SHIFTS:
        model.add(sum(works[p, d, s] for p in STAFF) == DEMAND[s])  # exact cover of every shift
    for p in STAFF:
        model.add_at_most_one(works[p, d, s] for s in SHIFTS)  # one shift per day
for p in STAFF:
    for d in range(DAYS - 1):
        model.add_implication(works[p, d, 'night'], works[p, d + 1, 'morning'].Not())  # rest after a night
    total = sum(works[p, d, s] for d in range(DAYS) for s in SHIFTS)
    model.add_linear_constraint(total, MIN_SHIFTS, MAX_SHIFTS)

# ## Soft wishes as penalties
#
# Working on a requested day off costs 10 points; each night shift above two per
# person costs 3, which spreads nights fairly.

penalties = []
for p, days in DAYS_OFF.items():
    penalties += [10 * works[p, d, s] for d in days for s in SHIFTS]
for p in STAFF:
    nights = sum(works[p, d, 'night'] for d in range(DAYS))
    excess = model.new_int_var(0, DAYS, f'{p}_extra_nights')
    model.add(excess >= nights - 2)
    penalties.append(3 * excess)
model.minimize(sum(penalties))

# ## Solve

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
solver.parameters.random_seed = 0
solver.parameters.num_workers = 4
status = solver.solve(model)
if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    raise SystemExit(f"No schedule satisfies the hard rules ({solver.status_name(status)}); relax DEMAND or limits")
print(f"Status {solver.status_name(status)}, penalty {solver.objective_value:.0f}, "
      f"best bound {solver.best_objective_bound:.0f}, {solver.wall_time:.1f} s")

# ## The rota

rota = pd.DataFrame(
    [[next((s[0].upper() for s in SHIFTS if solver.value(works[p, d, s])), '·') for d in range(DAYS)] for p in STAFF],
    index=STAFF, columns=[f'd{d + 1}' for d in range(DAYS)],
)
print(rota.to_string())
counts = pd.DataFrame({s: [sum(solver.value(works[p, d, s]) for d in range(DAYS)) for p in STAFF] for s in SHIFTS},
                      index=STAFF)
counts['total'] = counts.sum(axis=1)
print(counts)

broken = [(p, d + 1) for p, days in DAYS_OFF.items() for d in days if any(solver.value(works[p, d, s]) for s in SHIFTS)]
print(f"Day-off requests broken: {broken or 'none'}")

codes = {'·': 0, 'M': 1, 'E': 2, 'N': 3}
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(rota.replace(codes).to_numpy(dtype=float), cmap=plt.colormaps['Pastel1'].resampled(4), aspect='auto')
for (i, j), label in np.ndenumerate(rota.to_numpy()):
    ax.text(j, i, label, ha='center', va='center', fontsize=9)
ax.set(xticks=range(DAYS), xticklabels=rota.columns, yticks=range(len(STAFF)), yticklabels=STAFF,
       title='Two-week rota (M morning, E evening, N night)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shift_rota.png'), dpi=120)
plt.show()

rota.to_csv(os.path.join(OUTPUT_DIR, 'shift_rota.csv'))
with open(os.path.join(OUTPUT_DIR, 'shift_summary.json'), 'w') as f:
    json.dump({'status': solver.status_name(status), 'penalty': solver.objective_value, 'broken_requests': broken}, f)
print("Saved: shift_rota.png, shift_rota.csv, shift_summary.json")
