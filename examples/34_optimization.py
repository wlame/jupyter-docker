#!/usr/bin/env python3
"""
Optimization: Assignment, Vehicle Routing, Portfolios, and Production Planning
==============================================================================
Solves four classic operations-research problems with the tool that fits each:
an assignment problem twice (SciPy's Hungarian algorithm and a PuLP integer
program, which must agree), capacitated vehicle routing with OR-Tools, a
mean-variance portfolio frontier with CVXPY, and a production plan as a
mixed-integer program in Pyomo.

Solvers in this image: OR-Tools embeds CP-SAT, SCIP, CBC, HiGHS, GLOP, and
PDLP (reach them through `ortools.linear_solver.pywraplp` or `ortools.math_opt`);
CVXPY brings Clarabel, OSQP, and SCS, and uses SciPy's HiGHS for integer
problems; PuLP and Pyomo call the `cbc` command from the coinor-cbc system
package. The highspy package is left out on purpose: it cannot share a process
with OR-Tools (both ship a libhighs.so.1).

PuLP:      https://coin-or.github.io/pulp/
OR-Tools:  https://developers.google.com/optimization
CVXPY:     https://www.cvxpy.org/
Pyomo:     https://pyomo.readthedocs.io/
CBC:       https://github.com/coin-or/Cbc
"""

import json
import os
import shutil

import cvxpy as cp
import matplotlib
import numpy as np
import pulp
import pyomo.environ as pyo
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.optimize import linear_sum_assignment

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

if shutil.which('cbc') is None:
    raise SystemExit("the cbc solver command is missing; install the coinor-cbc package (it ships in this image)")

rng = np.random.default_rng(seed=0)
summary: dict[str, object] = {}

# =============================================================================
# Assignment: SciPy and a PuLP integer program must agree
# =============================================================================
print("=" * 60)
print("Assignment: 8 Workers, 8 Tasks")
print("=" * 60)

n_workers = 8
cost = rng.integers(10, 100, size=(n_workers, n_workers))

rows, cols = linear_sum_assignment(cost)
scipy_total = int(cost[rows, cols].sum())

problem = pulp.LpProblem('assignment', pulp.LpMinimize)
# PuLP 4 creates variables through the problem; keys are (worker, task) tuples.
assign = problem.add_variable_dict('assign', (range(n_workers), range(n_workers)), cat='Binary')
problem += pulp.lpSum(int(cost[w, t]) * assign[w, t] for w in range(n_workers) for t in range(n_workers))
for i in range(n_workers):
    problem += pulp.lpSum(assign[i, t] for t in range(n_workers)) == 1  # each worker gets one task
    problem += pulp.lpSum(assign[w, i] for w in range(n_workers)) == 1  # each task gets one worker
stats = problem.solve(pulp.COIN_CMD(msg=False))  # PuLP 4 returns solve statistics
pulp_total = int(pulp.value(problem.objective))

print(f"SciPy (Hungarian algorithm): total cost {scipy_total}")
print(f"PuLP + CBC (integer program): total cost {pulp_total}, status {stats.status.name}")
if scipy_total != pulp_total:
    raise SystemExit("assignment solvers disagree")
summary['assignment_cost'] = scipy_total

# =============================================================================
# Capacitated vehicle routing with OR-Tools
# =============================================================================
print("\n" + "=" * 60)
print("OR-Tools: Capacitated Vehicle Routing")
print("=" * 60)

n_customers, n_vehicles, capacity = 20, 4, 40
locations = np.vstack([[50, 50], rng.uniform(0, 100, size=(n_customers, 2))])  # index 0 is the depot
demands = [0, *rng.integers(3, 10, size=n_customers).tolist()]
distances = np.rint(np.linalg.norm(locations[:, None] - locations[None, :], axis=-1)).astype(int)

manager = pywrapcp.RoutingIndexManager(len(locations), n_vehicles, 0)
routing = pywrapcp.RoutingModel(manager)


def distance_callback(from_index: int, to_index: int) -> int:
    """Road distance between two routing indices (Euclidean, rounded)."""
    return int(distances[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])


def demand_callback(from_index: int) -> int:
    """Load picked up at a routing index."""
    return demands[manager.IndexToNode(from_index)]


routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_callback))
routing.AddDimensionWithVehicleCapacity(
    routing.RegisterUnaryTransitCallback(demand_callback), 0, [capacity] * n_vehicles, True, 'load'
)
search = pywrapcp.DefaultRoutingSearchParameters()
search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
search.time_limit.FromSeconds(3)
solution = routing.SolveWithParameters(search)
if solution is None:
    raise SystemExit("OR-Tools found no routing solution")

routes = []
for vehicle in range(n_vehicles):
    index, route = routing.Start(vehicle), []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(0)
    routes.append(route)
    load = sum(demands[node] for node in route)
    print(f"  vehicle {vehicle}: {len(route) - 2:2d} stops, load {load:2d}/{capacity}, route {route}")
total_distance = solution.ObjectiveValue()
print(f"Total distance: {total_distance} (demand {sum(demands)}, fleet capacity {capacity * n_vehicles})")
summary['routing_distance'] = total_distance

fig, ax = plt.subplots(figsize=(6, 6))
for vehicle, route in enumerate(routes):
    path = locations[route]
    ax.plot(path[:, 0], path[:, 1], '-o', ms=4, label=f'vehicle {vehicle}')
ax.plot(*locations[0], 'ks', ms=10, label='depot')
ax.set(title=f'OR-Tools routes, total distance {total_distance}', aspect='equal')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'optimization_routes.png'), dpi=120)
plt.close()
print("Saved: optimization_routes.png")

# =============================================================================
# Mean-variance portfolio frontier with CVXPY
# =============================================================================
print("\n" + "=" * 60)
print("CVXPY: Mean-Variance Efficient Frontier")
print("=" * 60)

n_assets = 8
expected_returns = rng.uniform(0.02, 0.15, size=n_assets)
factors = rng.normal(size=(n_assets, 3)) * 0.1
covariance = factors @ factors.T + np.diag(rng.uniform(0.01, 0.04, size=n_assets))

weights = cp.Variable(n_assets)
risk_aversion = cp.Parameter(nonneg=True)
portfolio_return = expected_returns @ weights
portfolio_risk = cp.quad_form(weights, covariance)
frontier_problem = cp.Problem(
    cp.Maximize(portfolio_return - risk_aversion * portfolio_risk),
    [cp.sum(weights) == 1, weights >= 0, weights <= 0.4],  # fully invested, long only, 40% cap
)
frontier = []
for gamma in np.logspace(-1, 2, 25):
    risk_aversion.value = gamma
    frontier_problem.solve()
    frontier.append((float(np.sqrt(portfolio_risk.value)), float(portfolio_return.value)))
risks, returns = np.array(frontier).T
print(f"Solver: {frontier_problem.solver_stats.solver_name}; 25 problems re-solved with a changing parameter")
print(f"Return range {returns.min():.3f} to {returns.max():.3f}; risk range {risks.min():.3f} to {risks.max():.3f}")
summary['frontier_points'] = len(frontier)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(risks, returns, 'o-', ms=3, label='efficient frontier')
ax.scatter(np.sqrt(np.diag(covariance)), expected_returns, c='crimson', label='single assets')
ax.set(xlabel='risk (standard deviation)', ylabel='expected return', title='Long-only portfolios, 40% cap')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'optimization_frontier.png'), dpi=120)
plt.close()
print("Saved: optimization_frontier.png")

# =============================================================================
# Production planning as a MILP in Pyomo, solved by CBC
# =============================================================================
print("\n" + "=" * 60)
print("Pyomo + CBC: Production Planning")
print("=" * 60)

products = {'chair': 45, 'table': 80, 'desk': 110, 'shelf': 30}  # profit per unit
hours = {  # resource hours per unit
    'chair': {'cutting': 1.0, 'assembly': 2.0, 'finishing': 1.0},
    'table': {'cutting': 2.0, 'assembly': 3.0, 'finishing': 2.0},
    'desk': {'cutting': 3.0, 'assembly': 4.0, 'finishing': 3.0},
    'shelf': {'cutting': 1.0, 'assembly': 1.0, 'finishing': 0.5},
}
available = {'cutting': 120, 'assembly': 200, 'finishing': 110}
setup_cost = 150  # paid once for each product line that runs at all

model = pyo.ConcreteModel()
model.P = pyo.Set(initialize=list(products))
model.R = pyo.Set(initialize=list(available))
model.make = pyo.Var(model.P, within=pyo.NonNegativeIntegers, bounds=(0, 60))
model.runs = pyo.Var(model.P, within=pyo.Binary)
model.profit = pyo.Objective(
    expr=sum(products[p] * model.make[p] - setup_cost * model.runs[p] for p in model.P), sense=pyo.maximize
)
model.capacity = pyo.Constraint(model.R, rule=lambda m, r: sum(hours[p][r] * m.make[p] for p in m.P) <= available[r])
model.setup = pyo.Constraint(model.P, rule=lambda m, p: m.make[p] <= 60 * m.runs[p])  # no output without setup

result = pyo.SolverFactory('cbc').solve(model)
plan = {p: int(round(pyo.value(model.make[p]))) for p in model.P}
print(f"Termination: {result.solver.termination_condition}")
for product, units in plan.items():
    print(f"  {product:6} {units:3d} units")
used = {r: sum(hours[p][r] * plan[p] for p in products) for r in available}
print("Hours used: " + ", ".join(f"{r} {used[r]:.0f}/{available[r]}" for r in available))
print(f"Profit after setup costs: {pyo.value(model.profit):,.0f}")
summary['production_plan'] = plan

with open(os.path.join(OUTPUT_DIR, 'optimization_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved: optimization_summary.json")
print("Done.")
