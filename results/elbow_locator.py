import numpy as np
from kneed import KneeLocator

def find_elbow_point(pareto_front_points):

    if not pareto_front_points:
        return None

    # Separate x and y
    x = np.array([p[0] for p in pareto_front_points])
    y = np.array([p[1] for p in pareto_front_points])

    try:
        kneedle = KneeLocator(x, y, curve='concave', direction='increasing')
        elbow_x = kneedle.knee
        if elbow_x is not None:
            # Get index of point closest to elbow_x
            idx = (np.abs(x - elbow_x)).argmin()
            return pareto_front_points[idx]
        else:
            return None
    except Exception as e:
        print(f"Error finding elbow point: {e}")
        return None