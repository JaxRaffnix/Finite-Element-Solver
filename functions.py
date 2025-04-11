import math
import numpy as np

def find_neighbor_index(mylist: np.array, start_point) -> tuple[list[int, int], float]:
    minimum_diff = math.inf
    neighbor = None

    # Iterate through the list to find the closest value to start_point
    for value in mylist:
        if value == start_point:  # Skip the start_point
            continue
        diff = abs(start_point - value)
        if diff < minimum_diff:
            minimum_diff = diff
            neighbor = value

    if neighbor is None:
        raise ValueError("No neighbor found.")

    # Return their index values and the closest distance
    start_index = np.where(mylist == start_point)[0][0]
    neighbor_index = np.where(mylist == neighbor)[0][0]
    return [int(start_index), int(neighbor_index)], minimum_diff