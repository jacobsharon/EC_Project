import numpy as np
from kneed import KneeLocator

###########################################################
# 1. find_elbow_point: Identify inflection point on curve #
###########################################################
def find_elbow_point(pareto_front_points):
    """
    Identifies the elbow point from a given set of Pareto front points using the KneeLocator algorithm.

    Parameters:
        pareto_front_points (list of tuple): A list of (complexity, F1) tuples representing the Pareto front.

    Returns:
        tuple or None: The (complexity, F1) coordinates of the detected elbow point,
                       or None if no elbow is found or if the input list is empty.
    """
    # 1.1 Return None immediately if the input list is empty
    if not pareto_front_points:
        return None

    # 1.2 Separate the complexity (x-axis) and F1 score (y-axis) values
    x = np.array([p[0] for p in pareto_front_points])  # Complexity values
    y = np.array([p[1] for p in pareto_front_points])  # Corresponding F1 scores

    try:
        # 1.3 Use the KneeLocator algorithm to detect the elbow
        #     The Pareto curve is assumed to be concave and increasing from left to right #
        kneedle = KneeLocator(x, y, curve='concave', direction='increasing')
        elbow_x = kneedle.knee

        # 1.4 If an elbow x-coordinate is found
        if elbow_x is not None:
            # 1.5 Identify the point on the curve closest to this elbow_x value
            idx = (np.abs(x - elbow_x)).argmin()
            return pareto_front_points[idx]  # Return the corresponding (complexity, F1) pair

        # 1.6 If the elbow could not be determined, return None
        else:
            return None

    # 1.7 Handle any errors that occur during knee detection
    except Exception as e:
        print(f"Error finding elbow point: {e}")
        return None