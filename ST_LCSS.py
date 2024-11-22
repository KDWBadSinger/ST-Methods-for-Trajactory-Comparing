# #------------------------------------------最初源代码---------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# from geopy.distance import geodesic
# from datetime import datetime
# from joblib import Parallel, delayed
#
# # Load the trajectory data
# file_path = 'test_data2.csv'
# data = pd.read_csv(file_path)
#
# # Convert place_id_dwell_max to datetime format
# data['place_id_dwell_max'] = pd.to_datetime(data['place_id_dwell_max'])
#
# # Helper function to calculate distance between two lat-lon points
# def calculate_distance(point1, point2):
#     return geodesic((point1['latitude'], point1['longitude']), (point2['latitude'], point2['longitude'])).meters
#
#
# # Define function to compute ST-LCSS and identify comovement segments
# def st_lcss_with_segments(trajectory1, trajectory2, max_time_diff=5 * 60, max_distance=50):
#     lcss_matrix = np.zeros((len(trajectory1), len(trajectory2)))
#     comovement_segments = []
#     current_segment = []
#
#     for i in range(len(trajectory1)):
#         for j in range(len(trajectory2)):
#             time_diff = abs(
#                 (trajectory1.iloc[i]['place_id_dwell_max'] - trajectory2.iloc[j]['place_id_dwell_max']).total_seconds())
#             distance = calculate_distance(trajectory1.iloc[i], trajectory2.iloc[j])
#
#             if time_diff <= max_time_diff and distance <= max_distance:
#                 if i == 0 or j == 0:
#                     lcss_matrix[i][j] = 1
#                 else:
#                     lcss_matrix[i][j] = max(lcss_matrix[i - 1][j - 1], lcss_matrix[i - 1][j], lcss_matrix[i][j - 1]) + 1
#
#                 # Add point to current segment if it's part of a comovement sequence
#                 if current_segment and (i - 1, j - 1) == current_segment[-1]:
#                     current_segment.append((i, j))
#                 else:
#                     # Save the previous segment and start a new one
#                     if current_segment:
#                         comovement_segments.append(current_segment)
#                     current_segment = [(i, j)]
#             else:
#                 if i == 0 and j == 0:
#                     lcss_matrix[i][j] = 0
#                 elif i == 0:
#                     lcss_matrix[i][j] = lcss_matrix[i][j - 1]
#                 elif j == 0:
#                     lcss_matrix[i][j] = lcss_matrix[i - 1][j]
#                 else:
#                     lcss_matrix[i][j] = max(lcss_matrix[i - 1][j], lcss_matrix[i][j - 1])
#
#     # Append the last segment if it exists
#     if current_segment:
#         comovement_segments.append(current_segment)
#
#     # Calculate the LCSS value
#     lcss_value = lcss_matrix[-1][-1]
#     return lcss_value, comovement_segments
#
#
# # Time window filter to avoid comparing trajectories with no overlapping time ranges
# def time_window_filter(traj1, traj2, max_time_diff=5 * 60):
#     min_time1, max_time1 = traj1['place_id_dwell_max'].min(), traj1['place_id_dwell_max'].max()
#     min_time2, max_time2 = traj2['place_id_dwell_max'].min(), traj2['place_id_dwell_max'].max()
#     return not (max_time1 < min_time2 - pd.Timedelta(seconds=max_time_diff) or max_time2 < min_time1 - pd.Timedelta(
#         seconds=max_time_diff))
#
#
# # Process each pair of trajectories, calculate LCSS and save comovement segments
# def process_trajectory_pair(traj1, traj2, advertiser1, advertiser2):
#     if not time_window_filter(traj1, traj2):
#         return None
#
#     lcss_value, comovement_segments = st_lcss_with_segments(traj1, traj2)
#     normalized_lcss_value = lcss_value / min(len(traj1), len(traj2))
#
#     # Store the comovement segments for output
#     segment_details = []
#     for segment in comovement_segments:
#         start_idx, end_idx = segment[0], segment[-1]
#         segment_details.append({
#             'Trajectory 1': advertiser1,
#             'Trajectory 2': advertiser2,
#             'Normalized ST-LCSS Value': normalized_lcss_value,
#             'User 1 Start Latitude': traj1.iloc[start_idx[0]]['latitude'],
#             'User 1 Start Longitude': traj1.iloc[start_idx[0]]['longitude'],
#             'User 1 Start Time': traj1.iloc[start_idx[0]]['place_id_dwell_max'],
#             'User 1 End Latitude': traj1.iloc[end_idx[0]]['latitude'],
#             'User 1 End Longitude': traj1.iloc[end_idx[0]]['longitude'],
#             'User 1 End Time': traj1.iloc[end_idx[0]]['place_id_dwell_max'],
#             'User 2 Start Latitude': traj2.iloc[start_idx[1]]['latitude'],
#             'User 2 Start Longitude': traj2.iloc[start_idx[1]]['longitude'],
#             'User 2 Start Time': traj2.iloc[start_idx[1]]['place_id_dwell_max'],
#             'User 2 End Latitude': traj2.iloc[end_idx[1]]['latitude'],
#             'User 2 End Longitude': traj2.iloc[end_idx[1]]['longitude'],
#             'User 2 End Time': traj2.iloc[end_idx[1]]['place_id_dwell_max']
#         })
#
#     return segment_details if segment_details else None
#
#
# # Group data by 'advertiser_id'
# trajectories = data.sort_values('place_id_dwell_max').groupby('advertiser_id')
#
# # Prepare pairs of trajectories for comparison
# advertiser_ids = list(trajectories.groups.keys())
# traj_pairs = [(trajectories.get_group(advertiser_ids[i]), trajectories.get_group(advertiser_ids[j]), advertiser_ids[i],
#                advertiser_ids[j])
#               for i in range(len(advertiser_ids)) for j in range(i + 1, len(advertiser_ids))]
#
# # Parallel processing of trajectory pairs
# comovement_segments = Parallel(n_jobs=6)(delayed(process_trajectory_pair)(traj1, traj2, advertiser1, advertiser2)
#                                           for traj1, traj2, advertiser1, advertiser2 in traj_pairs)
#
# # Flatten the list of lists and filter out None values
# comovement_segments = [segment for segments in comovement_segments if segments is not None for segment in segments]
#
# # Convert result to a DataFrame for output
# comovement_df = pd.DataFrame(comovement_segments)
# output_file_path = 'comovement_segments_with_lcss_testdata2.csv'
# comovement_df.to_csv(output_file_path, index=False)
# print(f"Comovement segments saved to {output_file_path}")
#
#
# # -----------------------------------------------融合李佳于代码11.21----------------------------------------------------------------------
#
# import pandas as pd
# import numpy as np
# from geopy.distance import geodesic
# from datetime import datetime, timedelta
# from joblib import Parallel, delayed
#
# # 文件路径
# file_path = '/home/dvw6ab/ST-LCSS/final_master_file.csv'
# output_file_path = 'final_comovement_segments_with_lcss.csv'
# chunk_size = 5000
#
# # 参数设置
# max_distance = 50  # 距离阈值（米）
# min_colocation_time = 300  # 最小共处时间（秒）
# read_limit = 25000  # 最大读取行数
# enable_read_limit = True  # 是否开启读取限制
#
# # 初始化输出文件表头
# pd.DataFrame(columns=[
#     'User1', 'User2', 'LCSS Value',
#     'Location1_Latitude', 'Location1_Longitude',
#     'Location2_Latitude', 'Location2_Longitude',
#     'Time1_Start', 'Time1_End',
#     'Time2_Start', 'Time2_End',
#     'Distance_m', 'Colocation_Time_Seconds'
# ]).to_csv(output_file_path, index=False)
#
# # Helper function to calculate haversine distance
# def haversine_distance(lat1, lon1, lat2, lon2):
#     return geodesic((lat1, lon1), (lat2, lon2)).meters
#
# # Calculate colocation time
# def calculate_colocation_time(min1, max1, min2, max2):
#     overlap_start = max(min1, min2)
#     overlap_end = min(max1, max2)
#     colocation_duration = (overlap_end - overlap_start).total_seconds()
#     return colocation_duration if colocation_duration > 0 else 0
#
# # Define LCSS calculation with matrix-based logic
# def st_lcss_with_segments(trajectory1, trajectory2, max_distance, min_colocation_time):
#     lcss_matrix = np.zeros((len(trajectory1), len(trajectory2)))
#     comovement_segments = []
#     current_segment = []
#
#     for i in range(len(trajectory1)):
#         for j in range(len(trajectory2)):
#             # 计算距离和共处时间
#             distance = haversine_distance(
#                 trajectory1.iloc[i]['latitude'], trajectory1.iloc[i]['longitude'],
#                 trajectory2.iloc[j]['latitude'], trajectory2.iloc[j]['longitude']
#             )
#             colocation_time = calculate_colocation_time(
#                 trajectory1.iloc[i]['place_id_dwell_min'], trajectory1.iloc[i]['place_id_dwell_max'],
#                 trajectory2.iloc[j]['place_id_dwell_min'], trajectory2.iloc[j]['place_id_dwell_max']
#             )
#
#             # 满足条件时更新 LCSS 矩阵
#             if distance <= max_distance and colocation_time >= min_colocation_time:
#                 if i == 0 or j == 0:
#                     lcss_matrix[i][j] = 1
#                 else:
#                     lcss_matrix[i][j] = max(lcss_matrix[i - 1][j - 1], lcss_matrix[i - 1][j], lcss_matrix[i][j - 1]) + 1
#
#                 # 更新共处段
#                 if current_segment and (i - 1, j - 1) == current_segment[-1]:
#                     current_segment.append((i, j))
#                 else:
#                     if current_segment:
#                         comovement_segments.append(current_segment)
#                     current_segment = [(i, j)]
#             else:
#                 # 保存最后的共处段
#                 if current_segment:
#                     comovement_segments.append(current_segment)
#                     current_segment = []
#
#     # 添加最后未保存的共处段
#     if current_segment:
#         comovement_segments.append(current_segment)
#
#     # 返回 LCSS 值和共处段
#     lcss_value = lcss_matrix[-1][-1]
#     return lcss_value, comovement_segments
#
# # Process each trajectory pair
# def process_trajectory_pair(traj1, traj2, user1, user2):
#     lcss_value, comovement_segments = st_lcss_with_segments(traj1, traj2, max_distance, min_colocation_time)
#     normalized_lcss_value = lcss_value / min(len(traj1), len(traj2)) if min(len(traj1), len(traj2)) > 0 else 0
#
#     results = []
#     for segment in comovement_segments:
#         for i, j in segment:
#             distance_m = haversine_distance(
#                 traj1.iloc[i]['latitude'], traj1.iloc[i]['longitude'],
#                 traj2.iloc[j]['latitude'], traj2.iloc[j]['longitude']
#             )
#             colocation_time = calculate_colocation_time(
#                 traj1.iloc[i]['place_id_dwell_min'], traj1.iloc[i]['place_id_dwell_max'],
#                 traj2.iloc[j]['place_id_dwell_min'], traj2.iloc[j]['place_id_dwell_max']
#             )
#             results.append({
#                 'User1': user1,
#                 'User2': user2,
#                 'LCSS Value': normalized_lcss_value,
#                 'Location1_Latitude': traj1.iloc[i]['latitude'],
#                 'Location1_Longitude': traj1.iloc[i]['longitude'],
#                 'Location2_Latitude': traj2.iloc[j]['latitude'],
#                 'Location2_Longitude': traj2.iloc[j]['longitude'],
#                 'Time1_Start': traj1.iloc[i]['place_id_dwell_min'],
#                 'Time1_End': traj1.iloc[i]['place_id_dwell_max'],
#                 'Time2_Start': traj2.iloc[j]['place_id_dwell_min'],
#                 'Time2_End': traj2.iloc[j]['place_id_dwell_max'],
#                 'Distance_m': distance_m,
#                 'Colocation_Time_Seconds': colocation_time
#             })
#     return results
#
# # Process data in chunks
# for chunk_index, chunk in enumerate(pd.read_csv(file_path, dtype={'latitude': 'float32', 'longitude': 'float32', 'advertiser_id': 'category'},
#                          chunksize=chunk_size)):
#     # 检查读取限制
#     if enable_read_limit and chunk_index * chunk_size >= read_limit:
#         break
#     # 对剩余部分进行行限制处理
#     if enable_read_limit and (chunk_index + 1) * chunk_size > read_limit:
#         chunk = chunk.iloc[:read_limit - chunk_index * chunk_size]
#
#     chunk['place_id_dwell_min'] = pd.to_datetime(chunk['place_id_dwell_min'], errors='coerce')
#     chunk['place_id_dwell_max'] = pd.to_datetime(chunk['place_id_dwell_max'], errors='coerce')
#     chunk = chunk.dropna(subset=['place_id_dwell_min', 'place_id_dwell_max'])
#
#     # Group by advertiser_id
#     trajectories = chunk.sort_values('place_id_dwell_min').groupby('advertiser_id')
#
#     # Prepare trajectory pairs
#     user_ids = list(trajectories.groups.keys())
#     traj_pairs = [
#         (trajectories.get_group(user_ids[i]), trajectories.get_group(user_ids[j]), user_ids[i], user_ids[j])
#         for i in range(len(user_ids)) for j in range(i + 1, len(user_ids))
#     ]
#
#     # Process pairs in parallel
#     results = Parallel(n_jobs=-1)(
#         delayed(process_trajectory_pair)(traj1, traj2, user1, user2)
#         for traj1, traj2, user1, user2 in traj_pairs
#     )
#
#     # Flatten results and write to output
#     flattened_results = [item for sublist in results if sublist for item in sublist]
#     if flattened_results:
#         pd.DataFrame(flattened_results).to_csv(output_file_path, mode='a', header=False, index=False)
#     print(f"Processed chunk {chunk_index + 1} of size {len(chunk)} and appended results to {output_file_path}")
#
# print(f"All chunks processed. Results saved to {output_file_path}")

#----------------------------修改后的原代码（修改了读取时间戳和lcss,效果很不错）-------------------------------------------------
# 基于改进1号代码，集成混合 LCSS 逻辑
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime
from joblib import Parallel, delayed

file_path = '/home/dvw6ab/ST-LCSS/final_master_file.csv'
output_file_path = 'final_comovement_segments_with_lcss.csv'

# Latitude and longitude bounds
LAT_MIN, LAT_MAX = 38.895, 38.907
LON_MIN, LON_MAX = -77.036, -77.012

# Distance and time thresholds
DISTANCE_THRESHOLD = 50  # meters
TIME_OVERLAP_THRESHOLD = pd.Timedelta(minutes=30)  # minimum overlap time


def calculate_distance(point1, point2):
    return geodesic((point1['latitude'], point1['longitude']), (point2['latitude'], point2['longitude'])).meters


# Filtering function for coordinates
def filter_by_coordinates(data, lat_min, lat_max, lon_min, lon_max, filter_by_coords):
    if filter_by_coords:
        return data[
            (data['latitude'] >= lat_min) & (data['latitude'] <= lat_max) &
            (data['longitude'] >= lon_min) & (data['longitude'] <= lon_max)
            ]
    return data


# Filtering function for time
def filter_by_month(data, month, filter_by_month):
    if filter_by_month:
        return data[(data['place_id_dwell_min'].dt.month == month) & (data['place_id_dwell_max'].dt.month == month)]
    return data


# Check time overlap
def calculate_time_overlap(start1, end1, start2, end2):
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_duration = overlap_end - overlap_start
    return overlap_duration if overlap_duration >= TIME_OVERLAP_THRESHOLD else pd.Timedelta(0)


# Define function to compute ST-LCSS and identify comovement segments
def st_lcss_with_segments(trajectory1, trajectory2, max_distance=DISTANCE_THRESHOLD):
    """
    Compute the Longest Common Subsequence (LCSS) with mixed logic for trajectory similarity.
    """
    lcss_matrix = np.zeros((len(trajectory1), len(trajectory2)))
    comovement_segments = []
    max_lcss_length = 0  # Record the longest LCSS length
    current_segment = []

    for i in range(len(trajectory1)):
        for j in range(len(trajectory2)):
            # Check time overlap and distance
            time_overlap = calculate_time_overlap(
                trajectory1.iloc[i]['place_id_dwell_min'], trajectory1.iloc[i]['place_id_dwell_max'],
                trajectory2.iloc[j]['place_id_dwell_min'], trajectory2.iloc[j]['place_id_dwell_max']
            )
            distance = calculate_distance(trajectory1.iloc[i], trajectory2.iloc[j])

            if time_overlap >= TIME_OVERLAP_THRESHOLD and distance <= max_distance:
                # Condition met: extend LCSS
                if i == 0 or j == 0:
                    lcss_matrix[i][j] = 1
                else:
                    lcss_matrix[i][j] = lcss_matrix[i - 1][j - 1] + 1

                # Update the current segment
                if current_segment and (i - 1, j - 1) == current_segment[-1]:
                    current_segment.append((i, j))
                else:
                    if current_segment:
                        comovement_segments.append(current_segment)
                    current_segment = [(i, j)]
            else:
                # Condition not met: break the sequence
                max_lcss_length = max(max_lcss_length, lcss_matrix[i - 1][j - 1])
                lcss_matrix[i][j] = 0

                # Save the current segment
                if current_segment:
                    comovement_segments.append(current_segment)
                    current_segment = []

    # Handle the last segment
    if current_segment:
        comovement_segments.append(current_segment)

    # Final LCSS value: longest segment from the matrix
    final_lcss_value = max(max_lcss_length, lcss_matrix[-1][-1])
    return final_lcss_value, comovement_segments


# Process each pair of trajectories, calculate LCSS and save comovement segments
def process_trajectory_pair(traj1, traj2, advertiser1, advertiser2):
    lcss_value, comovement_segments = st_lcss_with_segments(traj1, traj2)
    normalized_lcss_value = lcss_value / min(len(traj1), len(traj2)) if min(len(traj1), len(traj2)) > 0 else 0

    # Store the comovement segments for output
    segment_details = []
    for segment in comovement_segments:
        start_idx, end_idx = segment[0], segment[-1]
        segment_details.append({
            'Trajectory 1': advertiser1,
            'Trajectory 2': advertiser2,
            'Normalized ST-LCSS Value': normalized_lcss_value,
            'User 1 Start Latitude': traj1.iloc[start_idx[0]]['latitude'],
            'User 1 Start Longitude': traj1.iloc[start_idx[0]]['longitude'],
            'User 1 Start Time': traj1.iloc[start_idx[0]]['place_id_dwell_max'],
            'User 1 End Latitude': traj1.iloc[end_idx[0]]['latitude'],
            'User 1 End Longitude': traj1.iloc[end_idx[0]]['longitude'],
            'User 1 End Time': traj1.iloc[end_idx[0]]['place_id_dwell_max'],
            'User 2 Start Latitude': traj2.iloc[start_idx[1]]['latitude'],
            'User 2 Start Longitude': traj2.iloc[start_idx[1]]['longitude'],
            'User 2 Start Time': traj2.iloc[start_idx[1]]['place_id_dwell_max'],
            'User 2 End Latitude': traj2.iloc[end_idx[1]]['latitude'],
            'User 2 End Longitude': traj2.iloc[end_idx[1]]['longitude'],
            'User 2 End Time': traj2.iloc[end_idx[1]]['place_id_dwell_max']
        })

    return segment_details


# Write the header of the output file
with open(output_file_path, 'w') as f:
    pd.DataFrame(columns=[
        'Trajectory 1', 'Trajectory 2', 'Normalized ST-LCSS Value',
        'User 1 Start Latitude', 'User 1 Start Longitude', 'User 1 Start Time',
        'User 1 End Latitude', 'User 1 End Longitude', 'User 1 End Time',
        'User 2 Start Latitude', 'User 2 Start Longitude', 'User 2 Start Time',
        'User 2 End Latitude', 'User 2 End Longitude', 'User 2 End Time'
    ]).to_csv(f, index=False)

# Process data in chunks
chunk_size = 5000  # Adjust as needed for memory limits
processed_rows = 0
filter_by_coords = True  # Enable coordinate filtering
filter_by_month_flag = True  # Enable month filtering

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    processed_rows += len(chunk)

    # Convert timestamp column from UNIX seconds to datetime
    chunk['place_id_dwell_min'] = pd.to_datetime(chunk['place_id_dwell_min'], unit='s')
    chunk['place_id_dwell_max'] = pd.to_datetime(chunk['place_id_dwell_max'], unit='s')

    # Apply time filtering for July if enabled
    chunk = filter_by_month(chunk, month=7, filter_by_month=filter_by_month_flag)

    # Apply coordinate filtering
    chunk = filter_by_coordinates(chunk, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, filter_by_coords)
    print(f"Processing chunk with {len(chunk)} rows after filtering")

    # Group data by 'advertiser_id'
    trajectories = chunk.sort_values('place_id_dwell_max').groupby('advertiser_id')

    # Prepare pairs of trajectories for comparison
    advertiser_ids = list(trajectories.groups.keys())
    traj_pairs = [
        (trajectories.get_group(advertiser_ids[i]), trajectories.get_group(advertiser_ids[j]), advertiser_ids[i],
         advertiser_ids[j])
        for i in range(len(advertiser_ids)) for j in range(i + 1, len(advertiser_ids))]

    print(f"Number of trajectory pairs in this chunk: {len(traj_pairs)}")

    # Process pairs in parallel and write results incrementally
    results = Parallel(n_jobs=-1)(
        delayed(process_trajectory_pair)(traj1, traj2, advertiser1, advertiser2)
        for traj1, traj2, advertiser1, advertiser2 in traj_pairs
    )

    # Flatten and write results
    results = [segment for result in results if result for segment in result]
    if results:
        pd.DataFrame(results).to_csv(output_file_path, mode='a', header=False, index=False)
        print(f"Wrote {len(results)} rows to output file.")
    else:
        print("No valid results to write for this chunk.")

print(f"Processed a total of {processed_rows} rows.")

AATTGCGC
AATTCCGC


