import numpy as np
import matplotlib.pyplot as plt

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

def create_globaL_les(elements, boundary_condition):
    number_of_nodes = len(elements) +1
    
    coefficients_matrix = np.zeros(shape=(number_of_nodes, number_of_nodes))
    rhs_matrix = np.zeros(shape=(number_of_nodes, 1))

    # TODO: fix the out of bounds error!
    for element in elements:
        coefficients_matrix[element["Start Index"]][element["Start Index"]] += element["Coefficients"][0][0]
        coefficients_matrix[element["Start Index"]][element["End Index"]] += element["Coefficients"][0][1]
        coefficients_matrix[element["End Index"]][element["Start Index"]] += element["Coefficients"][1][0]
        coefficients_matrix[element["End Index"]][element["End Index"]] += element["Coefficients"][1][1]

        rhs_matrix[element["Start Index"]] += element["Right Hand Side"][0]
        rhs_matrix[element["End Index"]] += element["Right Hand Side"][1]

        # remove row from coefficients_matrix: 
    coefficients_matrix = np.delete(coefficients_matrix, boundary_condition["Lower Bound"]["x Index"], axis=0)
    coefficients_matrix = np.delete(coefficients_matrix, boundary_condition["Upper Bound"]["x Index"], axis=0)

    # remove row from right_hand_side_matrix
    rhs_matrix = np.delete(rhs_matrix, boundary_condition["Lower Bound"]["x Index"], axis=0)
    rhs_matrix = np.delete(rhs_matrix, boundary_condition["Upper Bound"]["x Index"], axis=0)

    rhs_matrix = rhs_matrix - boundary_condition["Upper Bound"]["Phi"] * coefficients_matrix[: ,boundary_condition["Upper Bound"]["x Index"]].reshape(number_of_nodes -2, 1) - boundary_condition["Lower Bound"]["Phi"] * coefficients_matrix[: ,boundary_condition["Lower Bound"]["x Index"]].reshape(number_of_nodes -2,1)

    # remove column from coefficients_matrix: 
    coefficients_matrix = np.delete(coefficients_matrix, boundary_condition["Lower Bound"]["x Index"], axis=1)
    coefficients_matrix = np.delete(coefficients_matrix, boundary_condition["Upper Bound"]["x Index"], axis=1)

    return coefficients_matrix, rhs_matrix

def solve_leq(coefficients_matrix, rhs_matrix, boundary_condition):
    reduced_solutions = np.linalg.solve(coefficients_matrix, rhs_matrix)

    # TODO: Insert the boundary values

    # TODO: sort the solution according to the node index
    pass

def plot_solution(y, boundary_condition):
    number_of_nodes = len(y)
    x = np.linspace(boundary_condition["Lower Bound"]["x"], boundary_condition["Upper Bound"]["x"], num=number_of_nodes)

    plt.plot(x, y)
    plt.scatter(x, y)

    plt.xlabel("x")
    plt.ylabel(r"\Phi (x)")
    plt.title(f"FEM Solution with {number_of_nodes} Nodes")

    plt.tight_layout()
    plt.show()