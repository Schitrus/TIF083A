# parse_script.py

import pandas as pd
import re

def parse_marker_data(filepath):
    """
    Parse time and positional data of motion capture markers from a Qualisys .tsv file.
    
    PARAMETERS
    ----------
    filepath : str, path object
    
    RETURNS
    -------
    markers_dict : dict
        Dictionary with marker names and their positional data.
    time : np.array
        Array of time values starting from zero.
    """
    with open(filepath, encoding='utf-8') as f:
        for _ in range(9):
            f.readline()
        title = f.readline().strip().split('\t')[1]
        marker_names = f.readline().strip().split('\t')[1:]
        collision = float(f.readline().strip().split('\t')[1])
        for _ in range(2):
            f.readline()
        
        data_df = pd.read_csv(filepath, sep='\t', skiprows=15, header=None)
    
    time = data_df.iloc[:, 1].to_numpy() - data_df.iloc[0, 1] # Extract and shift time data to start from 0
    markers_dict = {}
    masses = {}

    markers_dict['title'] = title
    markers_dict['collision'] = collision
    
    for i, marker in enumerate(marker_names):
        if not marker:
            continue
        
        marker = marker.strip()
        cm_match = re.match(r"([a-zA-Z])(\d+(_\d+)?)g$", marker)  # Match center of mass
        x_match = re.match(r"([a-zA-Z])x$", marker)               # Match x marker
        y_match = re.match(r"([a-zA-Z])y$", marker)               # Match y marker
        
        start_idx = 2 + 3 * i
        end_idx = start_idx + 3
        position_data = data_df.iloc[:, start_idx:end_idx].to_numpy()
        
        if cm_match:
            # Extract signifier and weight
            signifier = cm_match.group(1)
            mass_str = cm_match.group(2).replace("_", ".")  # Convert "82_4" to "82.4"
            mass_kg = round(float(mass_str) / 1000.0, 4)  # Convert grams to kilograms
            
            # Construct keys and populate masses dictionary
            prefix = f"{signifier}{mass_str}"  # e.g., "h82.4"
            key_name = f"{prefix}cm"         # e.g., "h82.4cm"
            markers_dict[key_name] = position_data
            masses[prefix] = mass_kg  # Store mass in kilograms
        
        elif x_match:
            prefix = x_match.group(1)
            key_name = f"{prefix}x"          # e.g., "hx"
            markers_dict[key_name] = position_data
        
        elif y_match:
            prefix = y_match.group(1)
            key_name = f"{prefix}y"          # e.g., "hy"
            markers_dict[key_name] = position_data

    return markers_dict, time, masses