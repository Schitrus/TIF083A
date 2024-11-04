# utils.py

import numpy as np

def calculate_velocity(position_data, time):
    """
    Calculate velocity as the numerical derivative of position over time.
    
    PARAMETERS
    ----------
    position_data : np.array
        Array of shape (N, 3) where N is the number of time points, and each row is (x, y, z).
    time : np.array
        Array of time values.
    
    RETURNS
    -------
    velocity : np.array
        Array of velocity values of shape (N, 3).
    """
    dt = np.diff(time)  # Time intervals
    velocity = np.diff(position_data, axis=0) / dt[:, None]  # Numerical derivative
    velocity = np.vstack([velocity, velocity[-1]])  # Pad to match original length
    return velocity

def calculate_momentum(velocity, mass):
    """
    Calculate linear momentum from velocity and mass.
    
    PARAMETERS
    ----------
    velocity : np.array
        Array of velocity values of shape (N, 3).
    mass : float
        Mass of the object.
    
    RETURNS
    -------
    momentum : np.array
        Array of momentum values of shape (N, 3).
    """
    return mass * velocity

def calculate_kinetic_energy(velocity, mass):
    """
    Calculate kinetic energy from velocity and mass.
    
    PARAMETERS
    ----------
    velocity : np.array
        Array of velocity values of shape (N, 3).
    mass : float
        Mass of the object.
    
    RETURNS
    -------
    kinetic_energy : np.array
        Array of kinetic energy values of shape (N,).
    """
    speed_squared = np.sum(velocity ** 2, axis=1)
    return 0.5 * mass * speed_squared

def is_1d_data(markers_dict):
    """
    Determine if the data is 1-dimensional based on the presence of only center of mass markers.
    
    PARAMETERS
    ----------
    markers_dict : dict
        Dictionary containing marker position data.
    
    RETURNS
    -------
    bool
        True if only center of mass markers are present (1D data), False otherwise.
    """
    return all("cm" in key for key in markers_dict.keys())
