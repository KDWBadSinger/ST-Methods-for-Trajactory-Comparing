import pandas as pd
import json

# Extract the desired columns
columns_to_keep = ['advertiser_id', 'latitude', 'longitude', 'place_id_dwell_max']

df = pd.read_csv('final_master_file.csv', usecols=columns_to_keep, nrows=25000)

# Convert 'place_id_dwell_max' to datetime format
df['place_id_dwell_max'] = pd.to_datetime(df['place_id_dwell_max'], unit='s')

# Save
df.to_csv('processed_trajectories_25k.csv', index=False)

print("Data has been processed and saved to the 'processed_trajectory_25k.csv' ")

# # generate trace for each user, sorted first by the timestamp 'place_id_dwell_max'
# user_trajectories = df.groupby('advertiser_id').apply(
#     lambda x: list(zip(
#         x.sort_values(by='place_id_dwell_max')['latitude'],
#         x.sort_values(by='place_id_dwell_max')['longitude'],
#         x.sort_values(by='place_id_dwell_max')['place_id_dwell_max'].astype(str)
#     ))
# )
#
# # Convert result to dict
# user_trajectories = user_trajectories.to_dict()
#
# # Save user trajectories into json
# with open('user_trajectories_5k.json', 'w') as f:
#     json.dump(user_trajectories, f)
#
# print("The user trajectories have been generated and saved to the 'user_trajectory_50k.json' file")