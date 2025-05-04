import numpy as np
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

def create_global_les(elements, boundary_condition):
    number_of_nodes = len(elements) +1
    number_of_boundaries = len(boundary_condition)
    
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

        
    boundary_indices = [boundary_condition["Lower Bound"]["x Index"], boundary_condition["Upper Bound"]["x Index"]]
    
    # remove rows
    coefficients_matrix = np.delete(coefficients_matrix, boundary_indices, axis=0)# remove row from coefficients_matrix: 
    rhs_matrix = np.delete(rhs_matrix, boundary_indices, axis=0) # remove row from right_hand_side_matrix

    # update rhs with boundary values
    rhs_matrix = rhs_matrix - boundary_condition["Upper Bound"]["Phi"] * coefficients_matrix[: ,boundary_indices[1]].reshape(number_of_nodes -number_of_boundaries, 1) - boundary_condition["Lower Bound"]["Phi"] * coefficients_matrix[: ,boundary_indices[0]].reshape(number_of_nodes -number_of_boundaries, 1)

    # remove column from coefficients_matrix: 
    coefficients_matrix = np.delete(coefficients_matrix, boundary_indices, axis=1)

    return coefficients_matrix, rhs_matrix

def solve_leq(coefficients_matrix, rhs_matrix, boundary_condition, number_of_nodes):

    reduced_solution = np.linalg.solve(coefficients_matrix, rhs_matrix)

    boundary_indices = [boundary_condition["Lower Bound"]["x Index"], boundary_condition["Upper Bound"]["x Index"]]
    boundary_values = [[boundary_condition["Lower Bound"]["Phi"]], 
                       [boundary_condition["Upper Bound"]["Phi"]]]
    kept_indices = np.setdiff1d(np.arange(number_of_nodes), boundary_indices)

    full_solution = np.full((number_of_nodes, 1), np.nan)
    full_solution[kept_indices] = reduced_solution
    full_solution[boundary_indices] = boundary_values

    return full_solution

def show_solution(y, nodes):
    plt.plot(nodes, y)
    plt.xlabel("x")
    plt.ylabel(r"$\Phi (x)$")
    plt.title(f"FEM Solution with {len(nodes)} Nodes")

    plt.tight_layout()
    plt.show()