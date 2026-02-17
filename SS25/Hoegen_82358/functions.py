# _____________________________________________________________________________
# General

# Author: Jan Hoegen

# Doc String Style Guide: NumPy

# _____________________________________________________________________________
# Library Imports


# system
import os
import warnings

# vectorized data
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, MatrixRankWarning

# dataframes
import pandas as pd

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# type hints
from numpy.typing import NDArray
from typing import Callable, Tuple, Union, Optional

sns.set_theme(style="white")


# _____________________________________________________________________________
# Helpers


def _get_centroid(arr):
    """returns (x_c, y_c). arr is of type [[x0, x1, x2, ...], [y0, y1, y2, ...]]"""

    return np.mean(arr, axis=0)


def get_number_of_elements(elements: np.ndarray) -> int:
    """Returns the number of elements in the list"""
    return len(elements)


def plot_knots_elements(knots: np.ndarray, elements=None, ax=None, x_label="x", y_label="y", title_suffix=None):
    """
    Plots the net and the elements defined in it.

    Args:
    - knots: Array of shape (n, 2), each row contains one knot. First column is x-coordinates, second column is y-coordinates.
    - elements: Array of shape (m, 3), each row describes one triangle element. First column is the index of the first knot, next column is the index of the second knot, and the last column is the index of the third knot.
    """
    LINEWIDTH = 0.4
    POINT_SIZE = 12  # size of knot markers
    TRIANGLE_COLOR = sns.color_palette("muted")[2]  # e.g., desaturated blue
    KNOT_COLOR = sns.color_palette("dark")[0]       # e.g., dark green

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # make x and y axes equal length
    # ax.set_aspect('equal', adjustable='box')

    # plot each triangle element
    if elements is not None and len(elements) > 0:
        for element in elements:
            triangle_coordinates = knots[element]
            polygon = mpl.patches.Polygon(
                triangle_coordinates, 
                closed=True, fill=False, 
                linewidth=LINEWIDTH, 
                facecolor=TRIANGLE_COLOR,
                edgecolor="black",
                alpha=0.5
            )    # create triangle from 3 points
            ax.add_patch(polygon)        # plot the triangles

    # plot the knots
    ax.scatter(
        knots[:,0], knots[:,1], 
        s=POINT_SIZE, 
        color=KNOT_COLOR,
        edgecolor='white',
        linewidth=0.3,
    )

    ax.set_title(f"Generated 2D Mesh with {len(knots)} nodes {title_suffix}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if ax is None:
        plt.tight_layout()
        plt.show()


def _estimate_missing_bc(stiffness_matrix):
    """stiffness_matrix: sp.lil_array"""

    print("Calculating missing constraints ...")
    stiffness_matrix = stiffness_matrix.toarray()

    rank = np.linalg.matrix_rank(stiffness_matrix)
    total_dof = stiffness_matrix.shape[0]

    return rank, total_dof


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
    area = np.abs( ( c[1] * b[0] - c[0] * b[1] ) / 2 )
    # if area <= 0:
    #     raise ValueError(f"Computed triangle area is non-positive: ({area}). Check knot ordering.")
    
    centroid = _get_centroid(knots[element])    # [x_M, y_M]

    # compute local matrices (coefficients) for the element
    B = np.outer(b, b) # outer product
    C = np.outer(c, c)
    D = np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ])
    stiffness_matrix = alpha1(*centroid) / (4 * area) * B  \
                     + alpha2(*centroid) / (4 * area) * C \
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
) -> Tuple[sp.lil_array, NDArray[np.float64]]:
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

    Note
    ----
    - the elemnt indices list has to of type int!
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
    global_stiffness = sp.lil_array((n_knots, n_knots), dtype=np.float64) # LIL format for efficient incremental assembly
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
    stiffness_matrix: sp.lil_array,
    load_vector: NDArray[np.float64],
    dirichlet_indices: NDArray[np.int_],
    dirichlet_func: Callable[[float, float], float],
) -> Tuple[sp.lil_array, NDArray[np.float64]]:
    """
    Apply Dirichlet boundary conditions by modifying the stiffness matrix and load vector.

    Parameters
    ----------
    knots : (n_knots, 2) float array
    dirichlet_indices : (n_b,) int array. list of indices from the knots variable where the dirichlet func is appropiate. The order (eg. clockwise, anticlockwise) is important.
    dirichlet : (x: float, y: float) -> float
    stiffness_matrix : dense or sparse matrix
    load_vector : (n,) array

    Returns
    -------
    reduced_matrix : matrix with boundary rows/cols removed
    reduced_rhs : adjusted load vector

    Note
    -----
    - The order of the dirichlet node indices matters!
    - make sure the indices list is of type int.
    ! This function has to be called AFTER inserting the robin boundary condition !
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
        solution = dirichlet_func(x, y)

        # stiffness matrix at i,i is 1
        stiffness_matrix.rows[index] = [index]
        stiffness_matrix.data[index] = [float(1)]

        # overwrite right hand side with solution.
        load_vector[index] = solution

    return stiffness_matrix, load_vector
    

def insert_robin_values(
    knots: np.ndarray,
    stiffness_matrix: sp.lil_array,
    load_vector: np.ndarray,
    robin_indices: NDArray[np.int_],
    robin_gamma: Callable[[float, float], float],
    robin_rhs: Callable[[float, float], float],
):
    """
    Inserts Robin boundary condition contributions into the global system.

    Parameters
    ----------
    knots : (n, 2) array of node coordinates
    stiffness_matrix : (n, n) LIL sparse matrix to modify
    load_vector : (n,) vector to modify
    robin_indices : list of [i, j] index pairs defining line segments on boundary
    robin_gamma : function rho(x, y) multiplying Φ in Robin condition
    robin_rhs : function f(x, y) on the boundary

    Note
    -----
    - make sure the indices list is of type int.
    ! This function has to be called BEFORE inserting the dirichlet boundary condition !
    """
    # --- Type checks ---
    if not isinstance(knots, np.ndarray):
        raise TypeError("knots must be a NumPy array")
    if not isinstance(stiffness_matrix, sp.lil_array):
        raise TypeError("stiffness_matrix must be a scipy.sparse.lil_array")
    if not isinstance(load_vector, np.ndarray):
        raise TypeError("load_vector must be a NumPy array")
    if not isinstance(robin_indices, np.ndarray):
        raise TypeError("robin_indices must be a NumPy array")
    if not np.issubdtype(robin_indices.dtype, np.integer):
        raise TypeError("robin_indices must contain integers")
    if not callable(robin_gamma):
        raise TypeError("robin_gamma must be callable")
    if not callable(robin_rhs):
        raise TypeError("robin_rhs must be callable")

    # --- Shape checks ---
    if knots.ndim != 2 or knots.shape[1] != 2:
        raise ValueError("knots must be of shape (n, 2)")
    n = knots.shape[0]
    if load_vector.shape != (n,):
        raise ValueError(f"load_vector must have shape ({n},)")
    if stiffness_matrix.shape != (n, n):
        raise ValueError(f"stiffness_matrix must have shape ({n}, {n})")
    if robin_indices.ndim != 2 or robin_indices.shape[1] != 2:
        raise ValueError("robin_indices must be of shape (m, 2)")
    
    # --- Optional: check that all indices are within bounds ---
    if np.any(robin_indices < 0) or np.any(robin_indices >= n):
        raise IndexError("robin_indices contain out-of-bound node indices")

    A = np.array([
        [2, 1],
        [1,2]
    ])
    B = np.array([1,1])

    
    for segment in robin_indices:
        length = np.linalg.norm(knots[segment[1]] - knots[segment[0]])

        centroid = _get_centroid(knots[segment])

        stiffness_matrix[np.ix_(segment, segment)] += length / 6 * robin_gamma(*centroid) * A

        load_vector[segment] += 0.5 * length * robin_rhs(*centroid) * B


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

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        try:
    
            if stiffness_matrix.shape[0] != load_vector.shape[0]:
                raise ValueError("Shape mismatch: stiffness_matrix and load_vector")

            stiffness_matrix_csr = stiffness_matrix.tocsr()
            solution_vector = spsolve(stiffness_matrix_csr, load_vector)

            return pd.DataFrame({
                "x" : knots[:,0],
                "y" : knots[:,1],
                "Phi" : solution_vector
            })
        except MatrixRankWarning:
            rank, total_dof = _estimate_missing_bc(stiffness_matrix)
            raise RuntimeError(f"Stiffness matrix is singular! Rank is {rank}, but there are {total_dof - rank} missing constraints. Probably missing dirichlet boundary conditions.")
        except Exception as e:
            raise RuntimeError(f"FEM solver failed: {e}")


# _____________________________________________________________________________
# Plot Solution


OUTPUT_FOLDER = "images"


def plot_result(
    elements: Union[np.ndarray, list],
    solution: pd.DataFrame,
    levels: int = 10,
    ax: Optional[plt.Axes] = None,
    suffix: Optional[str] = None
) -> None:
    """
    Plots the 2D FEM solution as a filled contour plot.

    Parameters
    ----------
    elements : array-like of shape (n_elements, 3)
        Triangle connectivity array where each row contains indices of three vertices.

    solution : pd.DataFrame
        DataFrame with columns "x", "y", and "Phi", representing node coordinates and computed potential.

    levels : int, optional
        Number of contour levels to use in the filled contour plot. Default is 10.

    ax : matplotlib.axes.Axes, optional
        An existing axis to plot into. If None, a new figure will be created.
    """
    if not isinstance(solution, pd.DataFrame):
        raise TypeError("solution must be a pandas DataFrame.")

    required_columns = {"x", "y", "Phi"}
    if not required_columns.issubset(solution.columns):
        raise ValueError(f"solution must contain the columns: {required_columns}")

    elements = np.asarray(elements)
    triangulation = mpl.tri.Triangulation(solution["x"], solution["y"], triangles=elements)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    contour = ax.tricontourf(triangulation, solution["Phi"], cmap="viridis", levels=levels)
    plt.colorbar(contour, ax=ax, label=r"$\Phi$")
    ax.set_title(f"Computed 2D FEM Solution with {len(solution)} nodes and {levels} color levels {suffix}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_aspect('equal', adjustable='box')

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename_solution = os.path.join(OUTPUT_FOLDER, "solution.png")
    plt.tight_layout()
    plt.savefig(filename_solution, dpi=300)
    plt.show()


def plot_comparison(
    elements: Union[np.ndarray, list],
    solution_df: pd.DataFrame,
    theoretical_filename: str,
    levels: int = 10,
    # ax_line: Optional[plt.Axes] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Plots the node-wise error between numerical and theoretical solution.

    Two plots are generated (unless axes are passed in):
    - Line plot: Error vs node index
    - Contour plot: Error field over 2D mesh

    Parameters
    ----------
    elements : array-like of shape (n_elements, 3)
        Triangle connectivity array.

    solution_df : pd.DataFrame
        DataFrame with columns "x", "y", and "Phi".

    theoretical_filename : str
        Path to a text file containing the theoretical Phi values.

    levels : int, optional
        Number of contour levels to use for the contour plot. Default is 10.

    ax_line : matplotlib.axes.Axes, optional
        Axis for line plot. If None, a new figure is created.

    ax_contour : matplotlib.axes.Axes, optional
        Axis for 2D contour plot. If None, a new figure is created.
    """
    if not isinstance(solution_df, pd.DataFrame):
        raise TypeError("solution_df must be a pandas DataFrame.")

    required_columns = {"x", "y", "Phi"}
    if not required_columns.issubset(solution_df.columns):
        raise ValueError(f"solution_df must contain the columns: {required_columns}")

    elements = np.asarray(elements)
    theoretical = np.loadtxt(theoretical_filename)

    if len(theoretical) != len(solution_df):
        raise ValueError("Theoretical data and solution must have the same length.")

    error = theoretical - solution_df["Phi"].values
    triangulation = mpl.tri.Triangulation(solution_df["x"], solution_df["y"], triangles=elements)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename_error_index = os.path.join(OUTPUT_FOLDER, "error_node.png")
    filename_error_2d = os.path.join(OUTPUT_FOLDER, "error_2d.png")

    # Line plot (Error per node)
    fig, ax_line = plt.subplots(figsize=(8, 6))
    ax_line.plot(np.arange(len(error)), error)
    ax_line.set_title("Solution Error for each Node")
    ax_line.set_xlabel("Node Index")
    ax_line.set_ylabel("Error")
    plt.tight_layout()
    plt.savefig(filename_error_index, dpi=300)
    plt.show()

    # 2D Error field
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.tricontourf(triangulation, error, cmap="viridis", levels=levels)
    plt.colorbar(contour, ax=ax, label="Error")
    ax.set_title("Error Field over 2D Domain")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax_contour.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(filename_error_2d, dpi=300)
    plt.show()
