# calculations.py

import numpy as np
from utils import calculate_velocity, calculate_momentum, calculate_kinetic_energy, is_1d_data

def calculate_angular_momentum(cm_position, edge_position, velocity, mass):
    """
    Calculate angular momentum based on center of mass and edge marker position.
    
    PARAMETERS
    ----------
    cm_position : np.array
        Array of center of mass position data of shape (N, 3).
    edge_position : np.array
        Array of edge marker position data of shape (N, 3).
    velocity : np.array
        Array of velocity values of shape (N, 3).
    mass : float
        Mass of the object.
    
    RETURNS
    -------
    angular_momentum : np.array
        Array of angular momentum values of shape (N, 3).
    """
    r = edge_position - cm_position  # Position vector from CM to edge
    p = mass * velocity  # Linear momentum
    return np.cross(r, p)  # Cross product for angular momentum

def compute_quantities(markers_dict, time, masses):
    """
    Compute momentum, kinetic energy, and angular momentum for each puck or rider.
    
    PARAMETERS
    ----------
    markers_dict : dict
        Dictionary containing marker position data.
    time : np.array
        Array of time values.
    masses : dict
        Dictionary of masses with keys matching CM keys in markers_dict.
    
    RETURNS
    -------
    results : dict
        Dictionary with calculated quantities for each puck or rider.
    """
    results = {}

    # Check if data is 1-dimensional
    data_is_1d = is_1d_data(markers_dict)

    for cm_key in masses.keys():
        cm_data = markers_dict.get(f"{cm_key}cm")
        
        if cm_data is None:
            continue  # Skip if no CM data
        
        # Calculate velocity and momentum
        velocity = calculate_velocity(cm_data, time)
        momentum = calculate_momentum(velocity, masses[cm_key])
        kinetic_energy = calculate_kinetic_energy(velocity, masses[cm_key])
        
        if not data_is_1d:
            # Calculate angular momentum only if data is 2D and edge markers are available
            x_data = markers_dict.get(f"{cm_key[0]}x")
            y_data = markers_dict.get(f"{cm_key[0]}y")
            
            if x_data is not None and y_data is not None:
                angular_momentum_x = calculate_angular_momentum(cm_data, x_data, velocity, masses[cm_key])
                angular_momentum_y = calculate_angular_momentum(cm_data, y_data, velocity, masses[cm_key])
                angular_momentum = angular_momentum_x + angular_momentum_y
            else:
                angular_momentum = None
        else:
            angular_momentum = None  # No angular momentum calculation for 1D data
        
        # Store results in dictionary
        results[cm_key] = {
            "momentum": momentum,
            "kinetic_energy": kinetic_energy,
            "angular_momentum": angular_momentum
        }
    
    return results
