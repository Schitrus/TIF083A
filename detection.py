# detection.py

import numpy as np
from scipy.signal import find_peaks

def calculate_velocity(position_data, time):
    """
    Calculate velocity as the numerical derivative of position over time.
    """
    dt = np.diff(time)  # Time intervals
    velocity = np.diff(position_data, axis=0) / dt[:, None]  # Numerical derivative
    velocity = np.vstack([velocity, velocity[-1]])  # Pad to match original length
    return velocity

# def detect_collision_times(markers_dict, time, threshold=0.1):
#     """
#     Detect collision start and end times based on sharp changes in CM velocity.
#     
#     PARAMETERS
#     ----------
#     markers_dict : dict
#         Dictionary containing marker position data.
#     time : np.array
#         Array of time values.
#     threshold : float, optional
#         Threshold for detecting significant acceleration peaks. Adjust based on data.
#     
#     RETURNS
#     -------
#     collision_times : list of tuples
#         List of tuples where each tuple contains (collision_start, collision_end).
#     """
#     collision_times = []
#     
#     for key, cm_data in markers_dict.items():
#         if 'cm' in key:  # Only consider center of mass data
#             # Step 1: Calculate velocity
#             velocity = calculate_velocity(cm_data, time)
#             speed = np.linalg.norm(velocity, axis=1)  # Magnitude of velocity
# 
#             # Step 2: Calculate acceleration (change in speed)
#             acceleration = np.abs(np.diff(speed) / np.diff(time))
#             
#             # Step 3: Identify peaks in acceleration
#             peaks, _ = find_peaks(acceleration, height=threshold)
#             
#             if len(peaks) == 0:
#                 continue  # No collision detected for this object
#             
#             for peak in peaks:
#                 # Step 4: Define a collision window around each peak
#                 # Adjust window size based on data and time resolution
#                 collision_start_idx = max(peak - 5, 0)
#                 collision_end_idx = min(peak + 5, len(time) - 1)
#                 
#                 collision_start = time[collision_start_idx]
#                 collision_end = time[collision_end_idx]
#                 
#                 collision_times.append((collision_start, collision_end))
#     
#     # Step 5: Combine overlapping collision windows (if multiple objects have similar times)
#     if collision_times:
#         collision_times = merge_collision_windows(collision_times)
#     
#     return collision_times

def detect_collision_times(markers_dict, time, threshold=0.1, min_speed=1e-3):
    """
    Detect collision start and end times based on changes in cm velocity.
    """
    collision_times = []
    
    for key, cm_data in markers_dict.items():
        if 'cm' in key:  # Center of mass data
            # Calculate velocity
            velocity = calculate_velocity(cm_data, time)
            speed = np.linalg.norm(velocity, axis=1)  # Magnitude of velocity

            # Filter out initial zero-speed data
            moving_indices = np.where(speed > min_speed)[0]
            if len(moving_indices) == 0:
                continue  # Skip if no movement detected
            
            # Truncate time and speed arrays to the period when objects are moving
            time = time[moving_indices[0]:]
            speed = speed[moving_indices[0]:]
            
            # Calculate acceleration (change in speed)
            acceleration = np.abs(np.diff(speed) / np.diff(time))
            
            # Detect peaks in acceleration
            peaks, _ = find_peaks(acceleration, height=threshold)
            
            if len(peaks) == 0:
                continue  # No collision detected for this object
            
            for peak in peaks:
                # Define a collision window around each peak
                collision_start_idx = max(peak - 5, 0)
                collision_end_idx = min(peak + 5, len(time) - 1)
                
                collision_start = time[collision_start_idx]
                collision_end = time[collision_end_idx]
                
                collision_times.append((collision_start, collision_end))
    
    # Combine overlapping collision windows
    if collision_times:
        collision_times = merge_collision_windows(collision_times)
    
    return collision_times


def merge_collision_windows(collision_times, merge_threshold=0.05):
    """
    Merge overlapping or nearby collision windows.
    
    PARAMETERS
    ----------
    collision_times : list of tuples
        List of collision time windows as (start, end) tuples.
    merge_threshold : float
        Time threshold for merging windows.
    
    RETURNS
    -------
    merged_windows : list of tuples
        List of merged collision time windows.
    """
    # Sort windows by start time
    collision_times.sort()
    
    merged_windows = []
    current_start, current_end = collision_times[0]
    
    for start, end in collision_times[1:]:
        if start - current_end <= merge_threshold:
            # Overlapping or close windows, extend the current window
            current_end = max(current_end, end)
        else:
            # No overlap, finalize the current window and start a new one
            merged_windows.append((current_start, current_end))
            current_start, current_end = start, end
    
    # Add the last window
    merged_windows.append((current_start, current_end))
    
    return merged_windows
