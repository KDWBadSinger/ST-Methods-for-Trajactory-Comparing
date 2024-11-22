from geopy.distance import geodesic
from scipy.spatial.distance import cdist
import numpy as np


# Trajectory segment function
def segment_trajectory(trajectory, distance_threshold):
    """
    Segment the given trajectory. If the distance between adjacent trajectory points exceeds the distance threshold,
     then segmented.

    :param trajectory: user trajectory，format [(latitude, longitude, timestamp), ...]
    :param distance_threshold: Distance threshold of the segment, in km

    :return: Segmented track, list of list form [[segment1], [segment2], ...]
    """
    segments = []
    current_segment = [trajectory[0]]  # Initializes the first track segment

    for i in range(1, len(trajectory)):
        prev_point = trajectory[i - 1][:2]  # Latitude and longitude of the previous point
        curr_point = trajectory[i][:2]  # Latitude and longitude of the current point
        distance = geodesic(prev_point, curr_point).km  # Calculate the distance between adjacent points

        # If the distance exceeds the threshold, segment it
        if distance > distance_threshold:
            segments.append(current_segment)  # Save current segment
            current_segment = [trajectory[i]]  # Start a new segment
        else:
            current_segment.append(trajectory[i])  # Adds to the current segment

    if current_segment:  # If there are unsaved segments, add them to the result
        segments.append(current_segment)

    return segments


# dynamic programming: Frechet Distance
def frechet_distance(P, Q):
    """
    Calculate the Frechet distance between two trajectories P and Q (implemented by dynamic programming).
    P and Q are two trajectories of the format [(latitude, longitude), ...]
    """
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)  # Initializes the cache matrix

    # Calculate Euclidean distance
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    # Dynamic programming fills the  ca  matrix
    ca[0, 0] = euclidean_distance(P[0], Q[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], euclidean_distance(P[i], Q[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], euclidean_distance(P[0], Q[j]))

    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                euclidean_distance(P[i], Q[j])
            )

    return ca[n - 1, m - 1]


# function: calculate the Frechet distance between two tracks
def calculate_frechet(trajectory1, trajectory2):
    """
    Take two trajectories and calculate the Frechet distance between them
    trajectory format [(latitude, longitude, timestamp), ...]
    """
    P = np.array([t[:2] for t in trajectory1])  # Only latitude and longitude are extracted
    Q = np.array([t[:2] for t in trajectory2])
    return frechet_distance(P, Q)
