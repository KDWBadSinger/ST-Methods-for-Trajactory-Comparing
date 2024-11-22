import json
import numpy as np
from function import segment_trajectory
from function import calculate_frechet

# ----------------------------------------------------------------------------------------------------------------------
# # Read the JSON file of the user trace
# with open('user_trajectories.json', 'r') as f:
#     user_trajectories = json.load(f)
#
# # Set the distance threshold for the track segment, in km
# distance_threshold = 0.2
#
# # Each user's trajectory is segmented
# user_segments = {user_id: segment_trajectory(trajectory, distance_threshold)
#                  for user_id, trajectory in user_trajectories.items()}
#
# # Check the segmentation results for a user
# for segment in user_segments['0002B079-2B7C-4E5F-AD0D-DB6F3D3AEED1']:
#     print(segment)
#
# # Save the segmented results
# with open('user_segments.json', 'w') as f:
#     json.dump(user_segments, f)
#
# print("轨迹分段完成并保存到 'user_segments.json' 文件中")
# ----------------------------------------------------------------------------------------------------------------------

# Read the segmented user trace JSON file
with open('user_segments.json', 'r') as f:
    user_segments = json.load(f)

# Extract all user ids
user_ids = list(user_segments.keys())

# Calculate the Frechet distance between two users
n = len(user_ids)
frechet_distances = np.zeros((n, n))

# Go through all the users and calculate the Frechet distance between the pairs
for i in range(n):
    for j in range(i + 1, n):
        # Select the first track of each user for comparison
        trajectory1 = user_segments[user_ids[i]][0]  # first track
        trajectory2 = user_segments[user_ids[j]][0]

        # Calculate the Frechet distance
        distance = calculate_frechet(trajectory1, trajectory2)
        frechet_distances[i, j] = distance
        frechet_distances[j, i] = distance  # symmetric matrix

# Print the Frechet distance matrix
print("Frechet Distance Matrix:")
print(frechet_distances)

# Save the Frechet distance matrix as a.npy file for later use
np.save('frechet_distances.npy', frechet_distances)