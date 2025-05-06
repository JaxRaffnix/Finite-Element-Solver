import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

def get_edge_indices(nodes: np.array) -> np.array:
    "Returns index pairs representing element edges based on nodes, sorted internally."
    sorted_knots = np.sort(nodes)
    knot_index_by_value = {value: index for index, value in enumerate(nodes)}

    indices = []
    for index in range(len(sorted_knots) -1):
        start_index = knot_index_by_value[sorted_knots[index]]
        stop_index = knot_index_by_value[sorted_knots[index +1]]
        indices.append([start_index, stop_index])

    indices = np.array(indices, dtype=int)
    return indices

def _get_element_coefficients(midpoint, length, alpha_func, beta_func):
    matrix1 = np.array([1, -1, -1, 1]).reshape(2,2)
    matrix2 = np.array([2, 1, 1, 2]).reshape(2,2)

    return alpha_func(midpoint) / length * matrix1 + length * beta_func(midpoint) / 6 * matrix2

def _get_element_rhs(midpoint: float, length: float, rhs_func):
    matrix = np.array([1,1])

    return length / 2 * rhs_func(midpoint) * matrix

def _get_element_matrices(start_value: float, end_value: float, alpha_func, beta_func, rhs_func):
    midpoint = (start_value + end_value) / 2
    length = abs(end_value - start_value)
    return _get_element_coefficients(midpoint, length, alpha_func, beta_func), _get_element_rhs(midpoint, length, rhs_func)

def assemble_element(nodes: np.array, start_index: float, end_index: float, alpha_func, beta_func, rhs_func):
    "Create the coefficient and right hand side matrix for a element defined by the index of its 2 edges."
    start_value = nodes[start_index]
    end_value = nodes[end_index]
    coefficients, rhs = _get_element_matrices(start_value, end_value, alpha_func, beta_func, rhs_func)
    element = {
        "Start Index": start_index,
        "End Index": end_index,
        "Coefficients": coefficients,
        "Right Hand Side": rhs
    }
    return element

def create_global_les(elements: list):
    number_of_nodes = len(elements) +1
    
    coefficients_matrix = np.zeros(shape=(number_of_nodes, number_of_nodes))
    rhs_matrix = np.zeros(shape=(number_of_nodes, 1))

    for element in elements:
        start_index = element["Start Index"]
        end_index = element["End Index"]

        coefficients_matrix[start_index][start_index] += element["Coefficients"][0][0]
        coefficients_matrix[start_index][end_index] += element["Coefficients"][0][1]
        coefficients_matrix[end_index][start_index] += element["Coefficients"][1][0]
        coefficients_matrix[end_index][end_index] += element["Coefficients"][1][1]

        rhs_matrix[start_index] += element["Right Hand Side"][0]
        rhs_matrix[end_index] += element["Right Hand Side"][1]

    return coefficients_matrix, rhs_matrix


def reduce_matrices(coefficients_matrix: np.array, rhs_matrix: np.array, boundary_condition: dict):
    """
    Reduces the system of equations for Dirichlet boundary conditions.
    
    Parameters:
        coefficients_matrix: Full system matrix (n x n)
        rhs_matrix: Right-hand side vector (n x 1)
        boundary_condition: Dictionary with boundary info. Entries must include "x Index" and optionally "Phi".
        
    Returns:
        (reduced_matrix, reduced_rhs): Modified system excluding Dirichlet DOFs.
    """
    boundary_indices = []
    
    # Apply Dirichlet values where Phi is defined
    for bc in boundary_condition.values():
        if "Phi" in bc:
            idx = bc["x Index"]
            phi = bc["Phi"]

            rhs_matrix -= phi * coefficients_matrix[:, idx].reshape(-1, 1)
            boundary_indices.append(idx)

    # Remove Dirichlet rows and columns
    coefficients_matrix = np.delete(coefficients_matrix, boundary_indices, axis=0)
    coefficients_matrix = np.delete(coefficients_matrix, boundary_indices, axis=1)
    rhs_matrix = np.delete(rhs_matrix, boundary_indices, axis=0)

    return coefficients_matrix, rhs_matrix


def add_robin_issue_values(coefficients_matrix: np.array, rhs_matrix: np.array, boundary_condition: dict):

    for bc in boundary_condition.values():
        if "Gamma" in bc:
            idx = bc["x Index"]
            gamma = bc["Gamma"]
            rho = bc["Rho"]

            coefficients_matrix[idx][idx] += gamma
            rhs_matrix[idx] += rho

    return coefficients_matrix, rhs_matrix


def solve_leq(coefficients_matrix: np.array, rhs_matrix: np.array):
    reduced_solution = np.linalg.solve(coefficients_matrix, rhs_matrix)
    return reduced_solution


def insert_boundary_values(reduced_solution: np.array, number_of_nodes: int, boundary_condition: dict):
    """
    Reconstructs the full solution by inserting known Dirichlet boundary values.

    Parameters:
        reduced_solution: Solution vector without Dirichlet nodes (shape: (n - num_dirichlet, 1))
        number_of_nodes: Total number of nodes in the original system
        boundary_condition: Dictionary of boundary conditions (must include "x Index" and optionally "Phi")

    Returns:
        full_solution: Complete solution vector including Dirichlet values (shape: (number_of_nodes, 1))
    """
    boundary_indices = []
    boundary_values = []

    for bc in boundary_condition.values():
        if "Phi" in bc:
            boundary_indices.append(bc["x Index"])
            boundary_values.append(bc["Phi"])

    boundary_indices = np.array(boundary_indices)
    boundary_values = np.array(boundary_values).reshape(-1, 1)  # Ensure column shape

    kept_indices = np.setdiff1d(np.arange(number_of_nodes), boundary_indices)

    full_solution = np.full((number_of_nodes, 1), np.nan)
    full_solution[kept_indices] = reduced_solution
    full_solution[boundary_indices] = boundary_values

    return full_solution


def create_solution_df(nodes: np.array, solution: np.array):
    solution_df = pd.DataFrame({"x": nodes.flatten(), "Phi": solution.flatten()})

    solution_df = solution_df.sort_values("x")
    return solution_df

def show_solution(y, nodes):
    plt.plot(nodes, y)
    plt.xlabel("x")
    plt.ylabel(r"$\Phi (x)$")
    plt.title(f"FEM Solution with {len(nodes)} Nodes")

    plt.tight_layout()
    plt.show()