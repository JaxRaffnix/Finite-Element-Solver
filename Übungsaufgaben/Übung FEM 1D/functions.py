import numpy as np
from nptyping import NDArray, Int, Float, Shape
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# _____________________________________________________________________________
# Global Settings

sns.set_theme()

# _____________________________________________________________________________
# Edge Generator

def get_edge_indices(nodes: NDArray[Shape["int, 1"], Float]) -> NDArray[Shape["int, 2"], Int]:
    """
    Generates index pairs representing element edges based on nodes in 1D.

    Arguments:
        nodes: 1D array of node coordinates/values.

    Returns:
        np.ndarray: An array of shape (n-1, 2) where each row contains the indices of the two nodes that form an edge.
    """
    if len(nodes) < 2:
        raise ValueError(f"At least two nodes are required to form an edge. Got length: {len(nodes)}")
    if not isinstance(nodes, np.ndarray):
        raise ValueError(f"Input nodes must be a numpy array. Got type: {type(nodes)}")
    if len(nodes.shape) != 1:
        raise ValueError(f"Input nodes must be a 1D array. Got shape: {format(nodes.shape)}")
    if np.unique(nodes).size != nodes.size:
        raise ValueError("Input nodes must be unique.")
    if not np.issubdtype(nodes.dtype, np.number):
        raise ValueError("Input nodes must be numeric.")

    sorted_indices = np.argsort(nodes)
    return np.column_stack((sorted_indices[:-1], sorted_indices[1:]))

# _____________________________________________________________________________
# Local Element Generator

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

def assemble_element(nodes: np.ndarray, start_index: int, end_index: int, alpha_func, beta_func, rhs_func):
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

# _____________________________________________________________________________
# Create Gloabal System

def create_global_les(elements: list):
    number_of_nodes = len(elements) +1
    
    coefficients_matrix = np.zeros(shape=(number_of_nodes, number_of_nodes))
    rhs_matrix = np.zeros(shape=(number_of_nodes, 1))

    for element in elements:
        start_index = element["Start Index"]
        end_index = element["End Index"]

        # TODO: dont manually index the coefficients, automate this
        coefficients_matrix[start_index][start_index] += element["Coefficients"][0][0]
        coefficients_matrix[start_index][end_index] += element["Coefficients"][0][1]
        coefficients_matrix[end_index][start_index] += element["Coefficients"][1][0]
        coefficients_matrix[end_index][end_index] += element["Coefficients"][1][1]

        rhs_matrix[start_index] += element["Right Hand Side"][0]
        rhs_matrix[end_index] += element["Right Hand Side"][1]

    return coefficients_matrix, rhs_matrix


def reduce_matrices(coefficients_matrix: np.ndarray, rhs_matrix: np.ndarray, boundary_condition: dict):
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

# _____________________________________________________________________________
# Add Boundary Conditions

def add_robin_issue_values(coefficients_matrix: np.ndarray, rhs_matrix: np.ndarray, boundary_condition: dict):

    for bc in boundary_condition.values():
        if "Gamma" in bc:
            idx = bc["x Index"]
            gamma = bc["Gamma"]
            rho = bc["Rho"]

            coefficients_matrix[idx][idx] += gamma
            rhs_matrix[idx] += rho

    return coefficients_matrix, rhs_matrix

# _____________________________________________________________________________
# Linear Solver

def solve_leq(coefficients_matrix: np.ndarray, rhs_matrix: np.ndarray):
    reduced_solution = np.linalg.solve(coefficients_matrix, rhs_matrix)
    return reduced_solution


def insert_boundary_values(reduced_solution: np.ndarray, number_of_nodes: int, boundary_condition: dict):
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


def create_solution_df(nodes: np.ndarray, solution: np.ndarray):
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