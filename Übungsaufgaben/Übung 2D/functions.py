# _____________________________________________________________________________
# General

# Author: Jan Hoegen

# Doc String Style Guide: NumPy

# _____________________________________________________________________________
# Library Imports


# system
import os

# vectorized data
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# dataframes
import pandas as pd

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# type hints
from numpy.typing import NDArray
from typing import Callable, Tuple, Union

sns.set_theme()


# _____________________________________________________________________________
# Helpers


def _get_centroid(arr):
    """returns (x_c, y_c)"""

    return np.mean(arr, axis=0)


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
        polygon = mpl.patches.Polygon(triangle_coordinates, closed=True, fill=False)    # create triangle from 3 points
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
    
    centroid = _get_centroid(knots[element])
    
    # centroid = np.mean(knots[element], axis=0)

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
) -> Tuple[sp.lil_matrix, NDArray[np.float64]]:
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

    # create global system
    n_knots = knots.shape[0]
    global_stiffness = sp.lil_matrix((n_knots, n_knots), dtype=np.float64) # LIL format for efficient incremental assembly
    global_load = np.zeros(n_knots, dtype=np.float64)

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
    dirichlet_indices: NDArray[np.int_],
    dirichlet: Callable[[float, float], float],
    stiffness_matrix: sp.lil_matrix,
    load_vector: NDArray[np.float64]
) -> Tuple[sp.lil_array, NDArray[np.float64]]:
    """
    Apply Dirichlet boundary conditions by modifying the stiffness matrix and load vector.

    Parameters
    ----------
    knots : (n_knots, 2) float array
    dirichlet_indices : (n_b,) int array. list of indices from the knots variable where the dirichlet func is appropiate.
    dirichlet : (x: float, y: float) -> float
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

    if not isinstance(dirichlet_indices, np.ndarray):
        raise TypeError("'dirichlet_indices' must be a numpy array.")
    if dirichlet_indices.ndim != 1:
        raise ValueError("'dirichlet_indices' must be a 1D array.")
    if not np.issubdtype(dirichlet_indices.dtype, np.integer):
        raise TypeError("'dirichlet_indices' must contain integers.")
    if np.any(dirichlet_indices < 0) or np.any(dirichlet_indices >= knots.shape[0]):
        raise ValueError("Some boundary indices are out of bounds.")

    if load_vector.ndim != 1:
        raise ValueError(f"'load_vector' must be a 1D array, got shape {load_vector.shape}.")
    if load_vector.shape[0] != knots.shape[0]:
        raise ValueError("Length of load vector must match number of knots.")

    if stiffness_matrix.shape[0] != stiffness_matrix.shape[1]:
        raise ValueError("Stiffness matrix must be square.")
    if stiffness_matrix.shape[0] != knots.shape[0]:
        raise ValueError("Stiffness matrix size must match number of knots.")
    
    # overwrite matrices at the indices of the known solutions
    # 1 * solution_i = load_i with load_i being the calculated solution.
    for index in dirichlet_indices:
        x,y = knots[index]
        solution = dirichlet(x, y)

        # stiffness matrix at i,i is 1
        stiffness_matrix.rows[index] = [index]
        stiffness_matrix.data[index] = [float(1)]

        # overwrite right hand side with solution.
        load_vector[index] = solution

    return stiffness_matrix, load_vector
    

def insert_robin_values(
    knots, 
    stiffness_matrix, 
    load_vector,
    robin_indices,
    robin_rho: Callable,
    ronbin_rhs: Callable,
    ):
    """
    robin indices : list of indices list. Each row describes one boundary segment. A row contains 2 values, each being the knot index that defines the segment.
    """
    A = np.array([
        [2, 1],
        [1,2]
    ])
    B = np.array([1,1])

    for segment in robin_indices:

        length = 1 # TODO: fix this!

        centroid = _get_centroid(knots[segment])

        stiffness_matrix[segment] += length / 6 * robin_rho(*centroid) * A
        load_vector[segment[0]] += 0.5 * length * ronbin_rhs(*centroid) * B


    return stiffness_matrix, load_vector
        


# _____________________________________________________________________________
# Solving


def solve_system(
    knots,
    stiffness_matrix: sp.lil_array,
    load_vector: NDArray[np.float64]
) -> pd.DataFrame:
    """
    Solve the linear FEM system stiffness_matrix * solution_vector = load_vector.

    Parameters
    ----------
    stiffness_matrix : (n, n) dense or sparse matrix
    load_vector : (n,) array

    Returns
    -------
    solution_vector: Dataframe with columns ['x', 'y', 'Phi']
    """
    stiffness_matrix.tocsr()

    if stiffness_matrix.shape[0] != load_vector.shape[0]:
        raise ValueError("Shape mismatch: stiffness_matrix and load_vector")

    solution_vector = spsolve(stiffness_matrix, load_vector)

    return pd.DataFrame({
        "x" : knots[:,0],
        "y" : knots[:,1],
        "Phi" : solution_vector
    })


def plot_result(elements, solution: pd.DataFrame, theoretical_filename = None, levels = 2):

    OUTPUT_FOLDER = "images"
    filename_mesh = os.path.join(OUTPUT_FOLDER, "mesh.png")
    filename_solution = os.path.join(OUTPUT_FOLDER, "solution.png")
    filename_error = os.path.join(OUTPUT_FOLDER, "error.png")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    triangulation = mpl.tri.Triangulation(solution["x"], solution["y"], triangles=elements)

    # Show Elements Mesh
    plt.figure(figsize=(8, 6))
    plt.triplot(triangulation, linewidth=0.5)
    plt.title(f"Mesh Structure with {len(elements)} Elements")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(filename_mesh, dpi=300)
    plt.show()

    # Show Solution
    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(triangulation, solution["Phi"], cmap="viridis", levels=levels)
    plt.colorbar(contour, label=r"$\Phi$")
    plt.title(f"Computed 2D FEM Solution with {len(solution)} nodes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(filename_solution, dpi=300)
    plt.show()

    # Check against Theoretical
    if theoretical_filename is None:
        return
    theoretical = np.loadtxt(theoretical_filename)
    error = theoretical - solution["Phi"]
    plt.plot(np.arange(len(error)), error)
    plt.title("Absolute Error")
    plt.xlabel("Index")
    plt.ylabel("Error per Node")
    plt.tight_layout()
    plt.savefig(filename_error, dpi=300)
    plt.show()

