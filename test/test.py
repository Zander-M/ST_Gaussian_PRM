"""
    code tests
"""
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import numpy as np

from ortools.init.python import init
from ortools.linear_solver import pywraplp
from ortools.graph.python import max_flow

def test_gaussian():
    """
        Test Gaussian KDE
    """
    # Generate some random data
    data = np.random.normal(0, 1, size=1000)

    # Fit a Gaussian KDE to the data
    kde = gaussian_kde(data)

    # Create a grid of points where we want to evaluate the KDE
    x_grid = np.linspace(-5, 5, 1000)

    # Evaluate the KDE on the grid
    kde_values = kde(x_grid)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_grid, kde_values, label='Gaussian KDE')
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram of data')
    plt.title('Gaussian Kernel Density Estimate')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("test.png")

def test_ortools():
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        print("can't create solver.")
        return

    x_var = solver.NumVar(0, 1, 'x')
    y_var = solver.NumVar(0, 2, 'y')
    print("# of variables", solver.NumVariables())

    infinity = solver.infinity()
    constraint = solver.Constraint(-infinity, 2, "ct")
    constraint.SetCoefficient(x_var, 1)
    constraint.SetCoefficient(y_var, 1)
    print("# of constraints", solver.NumConstraints())

    objective = solver.Objective()
    objective.SetCoefficient(x_var, 3)
    objective.SetCoefficient(y_var, 1)
    objective.SetMaximization()

    print(f"Solving with {solver.SolverVersion()}")
    result_status = solver.Solve()
    print(f"Status: {result_status}")
    if result_status != pywraplp.Solver.OPTIMAL:
        print("The problem does not have an optimal solution!")
        if result_status == pywraplp.Solver.FEASIBLE:
            print("A potentially suboptimal solution was found")
        else:
            print("The solver could not solve the problem.")
            return

    print("Solution:")
    print("Objective value =", objective.Value())
    print("x =", x_var.solution_value())
    print("y =", y_var.solution_value())

def test_max_flow():
    smf = max_flow.SimpleMaxFlow()

    # Define three parallel arrays: start_nodes, end_nodes, and the capacities
    # between each pair. For instance, the arc from node 0 to node 1 has a
    # capacity of 20.
    start_nodes = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3])
    end_nodes = np.array([1, 2, 3, 2, 4, 3, 4, 2, 4])
    capacities = np.array([20, 30, 10, 40, 30, 10, 20, 5, 20])
    all_arcs = smf.add_arcs_with_capacity(start_nodes, end_nodes, capacities)
    status =smf.solve(0, 4)

    if status != smf.OPTIMAL:
        print("There's an issue with the max flow")
        exit(1)
    print("Max flow:", smf.optimal_flow())
    print("")
    print(" Arc    Flow / Capacity")
    solution_flows = smf.flows(all_arcs)
    for arc, flow, capacity in zip(all_arcs, solution_flows, capacities):
        print(f"{smf.tail(arc)} / {smf.head(arc)}   {flow:3}  / {capacity:3}")
    print("Source side min-cut:", smf.get_source_side_min_cut())
    print("Sink side min-cut:", smf.get_sink_side_min_cut())

if __name__ == "__main__":
    # test_gaussian()
    # test_ortools()
    test_max_flow()
    pass