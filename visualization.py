# visualization.py

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(markers_dict, time, title="Object trajectories"):
    """
    Plot the x-y trajectories of objects from their center of mass data.
    """
    plt.figure(figsize=(10, 6))
    
    for key, data in markers_dict.items():
        if 'cm' in key:  # Plot only center of mass trajectories
            plt.plot(data[:, 0], data[:, 1], label=f"{key} (x-y path)")
    
    plt.xlabel("x position (mm)")
    plt.ylabel("y position (mm)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_quantity_over_time(results, time, quantity="momentum"):
    """
    Plot the specified quantity over time for each object.
    
    PARAMETERS
    ----------
    results : dict
        Dictionary containing calculated quantities for each object.
    time : np.array
        Array of time values.
    quantity : str
        The name of the quantity to plot ("momentum" or "kinetic_energy").
    """
    plt.figure(figsize=(10, 6))
    
    for obj, data in results.items():
        if data[quantity] is not None:
            if quantity == "momentum":
                # For vector quantities like momentum, compute the magnitude (norm)
                values = np.linalg.norm(data[quantity], axis=1)
                plt.ylabel("Momentum (kg·m/s)")
            elif quantity == "kinetic_energy":
                # For scalar quantities like kinetic energy, plot directly
                values = data[quantity]
                plt.ylabel("Kinetic energy (J)")
            else:
                continue  # Skip if an unsupported quantity is passed
            
            plt.plot(time, values, label=f"{obj} {quantity.capitalize()}")
    
    plt.xlabel("Time (s)")
    plt.title(f"{quantity.capitalize()} Over time")
    plt.legend()
    plt.grid()
    plt.show()


# def plot_quantity_over_time(results, time, quantity="momentum"):
#     """
#     Plot the specified quantity over time for each object.
#     """
#     plt.figure(figsize=(10, 6))
#     
#     for obj, data in results.items():
#         if data[quantity] is not None:
#             plt.plot(time, np.linalg.norm(data[quantity], axis=1), label=f"{obj} {quantity.capitalize()}")
#     
#     plt.xlabel("Time (s)")
#     plt.ylabel(f"{quantity.capitalize()} (kg·m/s)" if quantity == "momentum" else "Kinetic Energy (J)")
#     plt.title(f"{quantity.capitalize()} Over Time")
#     plt.legend()
#     plt.grid()
#     plt.show()

def identify_collision_times(time, start, end):
    """
    Identify indices for the start and end of the collision period.
    """
    start_idx = np.searchsorted(time, start)
    end_idx = np.searchsorted(time, end)
    return start_idx, end_idx

def calculate_totals(results, indices):
    """
    Calculate total momentum and kinetic energy over a specific time slice.
    """
    total_momentum = np.zeros(3)
    total_kinetic_energy = 0.0
    
    for obj, data in results.items():
        if data["momentum"] is not None:
            total_momentum += np.sum(data["momentum"][indices], axis=0)
        
        if data["kinetic_energy"] is not None:
            total_kinetic_energy += np.sum(data["kinetic_energy"][indices])
    
    return total_momentum, total_kinetic_energy

# def compare_before_after(results, time, collision_start, collision_end):
#     """
#     Calculate and compare total momentum and kinetic energy before and after a collision.
#     """
#     # Identify collision indices
#     start_idx, end_idx = identify_collision_times(time, collision_start, collision_end)
#     
#     # Calculate totals for before and after collision
#     before_indices = slice(start_idx - 10, start_idx)  # 10 frames before collision
#     after_indices = slice(end_idx, end_idx + 10)       # 10 frames after collision
#     
#     before_momentum, before_kinetic_energy = calculate_totals(results, before_indices)
#     after_momentum, after_kinetic_energy = calculate_totals(results, after_indices)
#     
#     # Calculate percentage differences
#     momentum_difference = np.linalg.norm(after_momentum - before_momentum) / np.linalg.norm(before_momentum) * 100
#     kinetic_energy_difference = abs(after_kinetic_energy - before_kinetic_energy) / before_kinetic_energy * 100
#     
#     # Display results
#     print("Before Collision:")
#     print("Total Momentum:", before_momentum)
#     print("Total Kinetic Energy:", before_kinetic_energy)
#     
#     print("\nAfter Collision:")
#     print("Total Momentum:", after_momentum)
#     print("Total Kinetic Energy:", after_kinetic_energy)
#     
#     print(f"\nMomentum Change: {momentum_difference:.2f}%")
#     print(f"Kinetic Energy Change: {kinetic_energy_difference:.2f}%")
#     
#     return momentum_difference, kinetic_energy_difference

def compare_before_after(results, time, collision_start, collision_end):
    """
    Calculate and compare total momentum and kinetic energy before and after a collision.
    """
    # Identify collision indices
    start_idx, end_idx = identify_collision_times(time, collision_start, collision_end)
    
    # Calculate totals for before and after collision
    before_indices = slice(start_idx - 10, start_idx)  # 10 frames before collision
    after_indices = slice(end_idx, end_idx + 10)       # 10 frames after collision
    
    before_momentum, before_kinetic_energy = calculate_totals(results, before_indices)
    after_momentum, after_kinetic_energy = calculate_totals(results, after_indices)
    
    # Calculate percentage differences
    if np.linalg.norm(before_momentum) != 0:
        momentum_difference = np.linalg.norm(after_momentum - before_momentum) / np.linalg.norm(before_momentum) * 100
    else:
        momentum_difference = np.inf
    
    if before_kinetic_energy != 0:
        kinetic_energy_difference = abs(after_kinetic_energy - before_kinetic_energy) / before_kinetic_energy * 100
    else:
        kinetic_energy_difference = np.inf
    
    # Display results
    print("Before collision:")
    print("Total momentum:", before_momentum)
    print("Total kinetic energy:", before_kinetic_energy)
    
    print("\nAfter collision:")
    print("Total momentum:", after_momentum)
    print("Total kinetic energy:", after_kinetic_energy)
    
    print(f"\nMomentum change: {momentum_difference:.2f}%" if momentum_difference != np.inf else "Momentum change: undefined")
    print(f"Kinetic energy change: {kinetic_energy_difference:.2f}%" if kinetic_energy_difference != np.inf else "Kinetic energy change: undefined")
    
    return momentum_difference, kinetic_energy_difference
