import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from numpy.typing import NDArray
from typing import Callable, Tuple


def get_number_of_elements(elements: np.ndarray) -> int:
    """Returns the number of elements in the list"""
    return elements.shape[0]

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
    if not isinstance(knots, np.ndarray):
        raise TypeError(f"'knots' must be a numpy ndarray, got {type(knots)}")
    if knots.ndim != 2 or knots.shape[1] != 2:
        raise ValueError(f"'knots' must have shape (n_knots, 2), got {knots.shape}")
    if knots.dtype not in [np.float32, np.float64]:
        raise TypeError(f"'knots' array must have dtype float32 or float64, got {knots.dtype}")

    if not isinstance(element, np.ndarray):
        raise TypeError(f"'element' must be a numpy ndarray, got {type(element)}")
    if element.ndim != 1 or element.shape[0] != 3:
        raise ValueError(f"'element' must be a 1D array of length 3, got shape {element.shape}")
    if not np.issubdtype(element.dtype, np.integer):
        raise TypeError(f"'element' array must contain integers, got {element.dtype}")

    if np.any(element < 0) or np.any(element >= len(knots)):
        raise ValueError(f"'element' contains indices out of range for knots array of length {len(knots)}")


    x = knots[element][:,0]
    y = knots[element][:,1]

    # Compute b and c vectors using vectorized operations
    b = [y[1] - y[2], y[2] - y[0], y[0] - y[1]]
    c = [x[2] - x[1], x[0] - x[2], x[1] - x[0]]

    # Area of the triangle element
    area = ( c[1] * b[0] - c[0] * b[1] ) / 2    
    if area <= 0:
        raise ValueError(f"Computed triangle area is non-positive ({area}). Check knot ordering.")
    
    centroid = np.mean(knots[element], axis=0)   # (x_c, y_c)

    # Matrices B and C formed by outer products of b and c vectors
    B = np.outer(b, b)
    C = np.outer(c, c)

    D = np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ])

    # compute local matrices for the element
    stiffness_matrix = 1/(area * 4) * alpha1(*centroid) * B \
                     + 1/(area * 4) * alpha2(*centroid) * C \
                     + (area / 12) * beta(*centroid) * D

    # RHS load vector (assuming rhs returns scalar source term)
    load_vector = (area / 3) * rhs(*centroid) * np.ones(3)

    return stiffness_matrix, load_vector

def assemble_global_system(knots, elements, alpha1, alpha2, beta, rhs):
    """
    """

    number_of_elements = get_number_of_elements(elements)

    global_stiffness = np.zeros([number_of_elements, number_of_elements])
    global_rhs = np.zeros([number_of_elements, number_of_elements])

    for i, element in enumerate(elements):
        if len(element) != 3:
            raise ValueError(f"Element {element} does not have exactly 3 knots.")
        
        local_stiffness, local_rhs = _calculate_local_stiffness(knots, element, alpha1, alpha2, beta, rhs)


        # Assemble global stiffness matrix and RHS vector
        global_stiffness[element[:, None], element] += local_stiffness
        global_rhs[element] += local_rhs