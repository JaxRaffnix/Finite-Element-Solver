# _____________________________________________________________________________
# Library Imports


# vectorized data
import numpy as np
from scipy.sparse import lil_matrix, csr_array, isspmatrix, issparse
from scipy.sparse.linalg import spsolve

# dataframes
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# type hints
from numpy.typing import NDArray
from typing import Callable, Tuple, Union

sns.set_theme()


# _____________________________________________________________________________
# Helpers


def get_number_of_elements(elements: np.ndarray) -> int:
    """Returns the number of elements in the list"""
    return len(elements)


def plot_knots_elements(knots: np.ndarray, elements=None, ax=None):
    """
    Plots the net and the elements defined in it.

    Args:
    - knots: Array of shape (n, 2), each row contains one knot. First column is x-coordinates, second column is y-coordinates.
    - elements: Array of shape (m, 3), each row describes one triangle element. First column is the index of the first knot, next column is the index of the second knot, and the last column is the index of the third knot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # make x and y axes equal length
    ax.set_aspect('equal', adjustable='box')

    # plot the knots
    ax.scatter(knots[:,0], knots[:,1])

    # highlight the start position
    ax.plot(knots[0,0], knots[0,1], color="green")

    if elements is None or len(elements) == 0:
        plt.show()
        print("No triangle elements to plot.")
        return

    # plot each triangle element
    for i, element in enumerate(elements):
        triangle_coordinates = knots[element]
        polygon = patches.Polygon(triangle_coordinates, closed=True, fill=False)    # create triangle from 3 points
        plt.gca().add_patch(polygon)        # plot the triangles
        centroid = np.mean(triangle_coordinates, axis=0)    # x, y
        plt.text(centroid[0], centroid[1], f"Tri {i}")

    plt.show()


# _____________________________________________________________________________
# Local Elements


def _calculate_local_stiffness(
    knots: NDArray[np.float64],
    element: NDArray[np.int_],
    alpha1: Callable[[float, float], float],
    alpha2: Callable[[float, float], float],
    beta: Callable[[float, float], float],
    rhs: Callable[[float, float], float]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the local stiffness matrix and load vector for a triangular 2D FEM element.

    Parameters:
    knots : NDArray[np.float64]
        Array of shape (n_knots, 2), where each row represents a knot coordinate:
        - first column is the x-coordinate (float)
        - second column is the y-coordinate (float)

    element : NDArray[np.int_]
        Array of shape (3,), containing indices of the knots forming the triangle element.
        Each value must be a valid index into the `knots` array.

    alpha1, alpha2, beta, rhs : Callable[[float, float], float]
        Scalar coefficient functions that take (x, y) coordinates and return a scalar float.

    Returns:
    --------
    stiffness_matrix : NDArray[np.float64]
        The (3, 3) local stiffness matrix for the element.

    load_vector : NDArray[np.float64]
        The (3,) local load vector for the element.
    
    Raises:
    -------
    TypeError
        If input types are not as expected.
    ValueError
        If array shapes do not match expected dimensions or indices are out of range.
    """
    # Validate types and shapes
    if not isinstance(element, np.ndarray):
            raise TypeError(f"'element' must be a numpy ndarray, got {type(element)}")
    if element.ndim != 1 or element.shape[0] != 3:
        raise ValueError(f"'element' must be a 1D array of length 3, got shape {element.shape}")
    if not np.issubdtype(element.dtype, np.integer):
        raise TypeError(f"'element' array must contain integers, got {element.dtype}")
    if np.any(element < 0) or np.any(element >= int(len(knots))):
        raise ValueError(f"'element' contains indices out of range for knots array of length {len(knots)}")

    # list of x, y coordinates for all knots of the element
    x, y = knots[element].T
    
    # Compute helper vetors b and c using vectorized operations
    b = np.array([
        y[1] - y[2], 
        y[2] - y[0], 
        y[0] - y[1]
    ])
    c = np.array([
        x[2] - x[1], 
        x[0] - x[2], 
        x[1] - x[0]
    ])

    # Area of the triangle element
    area = ( c[1] * b[0] - c[0] * b[1] ) / 2    
    if area <= 0:
        raise ValueError(f"Computed triangle area is non-positive: ({area}). Check knot ordering.")
    
    centroid = np.mean(knots[element], axis=0)   # returns (x_c, y_c)

    # compute local matrices (coefficients) for the element
    B = np.outer(b, b) # outer product
    C = np.outer(c, c)
    D = np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ])
    stiffness_matrix = alpha1(*centroid) / (4 * area) * B  \
                     + alpha2(*centroid) / (4 * area)  * C \
                     + beta(*centroid) * area / 12 * D

    # compute local load (right hand side) vector 
    E = np.array([1, 1, 1])
    load_vector = rhs(*centroid) * area / 3 * E

    return stiffness_matrix, load_vector


# _____________________________________________________________________________
# global system


def assemble_global_system(
    knots: NDArray[np.float64],
    elements: NDArray[np.int_],
    alpha1: Callable[[float, float], float],
    alpha2: Callable[[float, float], float],
    beta: Callable[[float, float], float],
    rhs: Callable[[float, float], float]
) -> Tuple[lil_matrix, NDArray[np.float64]]:
    """
    Assemble the global stiffness matrix and load vector from all triangular elements.

    Parameters
    ----------
    knots : (n_knots, 2) array of floats
        Coordinates of all knots with x,y value per column.

    elements : (n_elements, 3) array of ints
        Indices of knots forming each triangular element. Each triangle is formed by 3 points. Each row describes one element.

    alpha1, alpha2, beta, rhs : callable
        Functions of (x, y) returning scalars.

    Returns
    -------
    global_stiffness : (n_knots, n_knots) array of floats
        The assembled global stiffness matrix.

    global_load : (n_knots,) array of floats
        The assembled global load vector.
    """
    # Validate types and shapes
    if not isinstance(knots, np.ndarray):
        raise TypeError(f"'knots' must be a numpy ndarray, got {type(knots)}")
    if knots.ndim != 2 or knots.shape[1] != 2:
        raise ValueError(f"'knots' must have shape (n_knots, 2), got {knots.shape}")
    if knots.dtype not in [np.float32, np.float64]:
        raise TypeError(f"'knots' array must have dtype float32 or float64, got {knots.dtype}")

    # create global system matrices
    n_knots = knots.shape[0]
    global_stiffness = lil_matrix((n_knots, n_knots), dtype=np.float64) # LIL format for efficient incremental assembly
    global_load = np.zeros(n_knots, dtype=np.float64)

    # Loop over all elements and assemble global system
    for element in elements: 
        local_stiffness, local_load = _calculate_local_stiffness(knots, element, alpha1, alpha2, beta, rhs)

        # add contributions to global matrix and vector
        global_load[element] += local_load
        # np.ix_(element, element) creates a 3x3 index mesh to target the global submatrix
        global_stiffness[np.ix_(element, element)] += local_stiffness
        
    # Convert to CSR format for efficient solver use
    return global_stiffness, global_load


# _____________________________________________________________________________
# Boundary Conditions


def insert_dirichlet_values(
    knots: NDArray[np.float64],
    boundary_indices: NDArray[np.int_],
    boundary_func: Callable[[float, float], float],
    stiffness_matrix: lil_matrix,
    load_vector: NDArray[np.float64]
) -> Tuple[csr_array, NDArray[np.float64]]:
    """
    Apply Dirichlet boundary conditions by modifying the stiffness matrix and load vector.

    Parameters
    ----------
    knots : (n_knots, 2) float array
    boundary_indices : (n_b,) int array
    boundary_func : (x: float, y: float) -> float
    stiffness_matrix : dense or sparse matrix
    load_vector : (n,) array

    Returns
    -------
    reduced_matrix : matrix with boundary rows/cols removed
    reduced_rhs : adjusted load vector
    """
    # Validate types and shapes
    if not isinstance(knots, np.ndarray):
        raise TypeError("'knots' must be a numpy array.")
    if knots.ndim != 2 or knots.shape[1] != 2:
        raise ValueError(f"'knots' must have shape (n, 2), got {knots.shape}.")
    if not np.issubdtype(knots.dtype, np.floating):
        raise TypeError(f"'knots' must have float dtype, got {knots.dtype}.")

    if not isinstance(boundary_indices, np.ndarray):
        raise TypeError("'boundary_indices' must be a numpy array.")
    if boundary_indices.ndim != 1:
        raise ValueError("'boundary_indices' must be a 1D array.")
    if not np.issubdtype(boundary_indices.dtype, np.integer):
        raise TypeError("'boundary_indices' must contain integers.")
    if np.any(boundary_indices < 0) or np.any(boundary_indices >= knots.shape[0]):
        raise ValueError("Some boundary indices are out of bounds.")

    if load_vector.ndim != 1:
        raise ValueError(f"'load_vector' must be a 1D array, got shape {load_vector.shape}.")
    if load_vector.shape[0] != knots.shape[0]:
        raise ValueError("Length of load vector must match number of knots.")

    if stiffness_matrix.shape[0] != stiffness_matrix.shape[1]:
        raise ValueError("Stiffness matrix must be square.")
    if stiffness_matrix.shape[0] != knots.shape[0]:
        raise ValueError("Stiffness matrix size must match number of knots.")
    
    for index in boundary_indices:
        x,y = knots[index]
        solution = boundary_func(x, y)

        stiffness_matrix.rows[index] = [index]
        stiffness_matrix.data[index] = [float(1)]

        load_vector[index] = solution

    return stiffness_matrix.tocsr(), load_vector
    
    # x_b = knots[boundary_indices][:, 0]
    # y_b = knots[boundary_indices][:, 1]
    # g_b = boundary_func(x_b, y_b)

    # rhs = load_vector.copy()

    # # --- Modify RHS using Dirichlet values ---
    # for i, idx in enumerate(boundary_indices):
    #     col = stiffness_matrix[:, idx]
    #     if isspmatrix(col):
    #         col = col.toarray().ravel()
    #     else:
    #         col = np.asarray(col).ravel()

    #     rhs -= col * g_b[i]

    # # --- Remove Dirichlet DOFs ---
    # keep = np.setdiff1d(np.arange(stiffness_matrix.shape[0]), boundary_indices)

    # if isspmatrix(stiffness_matrix):
    #     stiffness_matrix = stiffness_matrix.tocsr()
    #     reduced_matrix = stiffness_matrix[keep][:, keep]
    # else:
    #     reduced_matrix = stiffness_matrix[np.ix_(keep, keep)]

    # reduced_rhs = rhs[keep]

    # return reduced_matrix, reduced_rhs


# _____________________________________________________________________________
# Solving


def solve_fem_system(
    stiffness_matrix: csr_array,
    load_vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Solve the linear FEM system A * u = b.

    Parameters
    ----------
    stiffness_matrix : (n, n) dense or sparse matrix
    load_vector : (n,) array

    Returns
    -------
    solution : (n,) array
        Solution vector u.
    """

    if issparse(stiffness_matrix):
        return spsolve(stiffness_matrix, load_vector)
    else:
        return np.linalg.solve(stiffness_matrix, load_vector)
    

# def add_known_boundary_values(knots, solutions_vector, boundary_incides, boundary_func):
#     result_indices = []
#     results = np.zeros(len(knots))

#     for index in boundary_incides:
#         results[index] = boundary_func(*knots[index])
#         result_indices.append(index)

#     result_indices = np.array(result_indices)
#     results = np.array(results).reshape(-1, 1)  # Ensure column shape

#     kept_indices = np.setdiff1d(np.arange(len(knots)), result_indices)

#     full_solution = np.full((len(knots), 1), np.nan)
#     full_solution[kept_indices] = solutions_vector.reshape(-1, 1)
#     full_solution[result_indices] = results[result_indices].reshape(-1, 1)

#     return full_solution

def _create_solution_df(nodes: np.ndarray, solution: np.ndarray) -> pd.DataFrame:
    """
    Create a DataFrame with node coordinates and solution values.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    solution : (n_nodes,) or (n_nodes, 1) array

    Returns
    -------
    DataFrame with columns ['x', 'y', 'Phi']
    """
    if solution.ndim == 2 and solution.shape[1] == 1:   
        solution = solution.ravel()

    return pd.DataFrame({
        "x": nodes[:, 0],
        "y": nodes[:, 1],
        "Phi": solution
    })


# _____________________________________________________________________________
# Show Results


def compare_results_2d(nodes: np.ndarray, solutions_file: str, own_solution: np.ndarray):
    """
    Compare 2D FEM solution to theoretical data and plot:
    - Theoretical solution (2D)
    - Computed solution (2D)
    - Absolute error (2D)
    - Absolute error vs. node index (1D)

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    solutions_file : str
        Path to .txt file with theoretical solution.
    own_solution : (n_nodes,) or (n_nodes, 1) array
    """
    theoretical = np.loadtxt(solutions_file)
    if theoretical.ndim == 2:
        theoretical = theoretical.ravel()
    if own_solution.ndim == 2:
        own_solution = own_solution.ravel()

    error = np.abs(own_solution - theoretical)

    df_theoretical = _create_solution_df(nodes, theoretical)
    df_computed = _create_solution_df(nodes, own_solution)
    df_error = _create_solution_df(nodes, error)

    fig = plt.figure(figsize=(16, 8))
    axs = fig.subplots(2, 2)

    contour_data = [
        (df_theoretical, "Theoretical Solution", "viridis"),
        (df_computed, "Computed Solution", "viridis"),
        (df_error, "Absolute Error (2D)", "inferno"),
    ]

    for ax, (df, title, cmap) in zip(axs.flat[:3], contour_data):
        tpc = ax.tricontourf(df["x"], df["y"], df["Phi"], levels=20, cmap=cmap)
        fig.colorbar(tpc, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    # 1D plot of absolute error vs. node index
    axs.flat[3].plot(np.arange(len(error)), error, color="darkred", marker="o", linestyle="-", markersize=3)
    axs.flat[3].set_title("Absolute Error per Node")
    axs.flat[3].set_xlabel("Node Index")
    axs.flat[3].set_ylabel("|Error|")
    axs.flat[3].grid(True)

    plt.suptitle("Comparison of Solutions and Error", fontsize=16)
    plt.show()