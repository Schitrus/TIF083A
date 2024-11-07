import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from parse_script import parse_marker_data

# Precalculated moment of inertias
moment_of_inertia = {"5.9": 3.084225e-6, "14.6": 4.5625e-6, "27.7": 8.65625e-6, "31.6": 0.00001315025, "34.0": 6.8e-6, \
                     "34.3": 6.83e-6, "22.9": 0.0000152542625, "82.2": 0.000038233275, "82.4": 0.0000383263, \
                        "28.3": 0.0000092010375, "43.5": 0.00008632575, "43.3": 0.0000845703125, "396.1": 0.0003083054}

def generate_animation(filepath, speed=1.0, traceback_time=1.0, start=0.0, end=1.0):
    """Generates animation from the data of the experiment
    
    Arguments:
    filepath -- The filepath to the tsv file
    rmm -- if angular momentum should be tracked (must be available in data)
    speed -- The time speed of the animation, higher is faster
    traceback_time -- How long back in time (s) the objects should be traced with a path
    start - The start time of the animation from the data as percentage e.g. 0.3 = 30%
    end - The end time of the animation from the data as percentage, must be greater than start and no greater than 1.0
    """
    # Read data
    markers_dict, ts, masses = parse_marker_data(filepath)
    filename = filepath.split('/')[-1].split('.')[0]

    # Camera speed is 500Hz
    camera_speed = 500
    actual_speed = camera_speed * speed

    # Goal to have 50 fps in animation, more is buggy
    fps = 50
    stepsize=max(actual_speed//fps, 1) # How many steps to go for each frame

    frame_count = len(ts)

    fig = plt.figure()
    fig.set_dpi(180)
    fig.set_figheight(10)
    ax = fig.add_subplot(2, 1, 1)
    gx = fig.add_subplot(2, 1, 2)

    # Parameters for graph
    ####################
    # sample step size
    sample_stepsize = 10

    # Set frequency and total time
    frequency = 500/sample_stepsize
    ts0 = np.array(ts[::sample_stepsize][:-1]) - ts[0]

    # Point to calculate angular momentum relative
    point = np.array([0.45,0.45,0])

    # Determines the scale of the figure
    x_min = np.min([np.min(pos.T[0]) for pos in markers_dict.values()])
    x_max = np.max([np.max(pos.T[0]) for pos in markers_dict.values()])
    y_min = np.min([np.min(pos.T[1]) for pos in markers_dict.values()])
    y_max = np.max([np.max(pos.T[1]) for pos in markers_dict.values()])

    # Predetermined unique colors
    colors = ['cornflowerblue', 'crimson', 'green', 'coral', 'darkorchid', 
              'hotpink', 'lightseagreen', 'sandybrown', 'springgreen', 'violet',
              'tomato', 'darkolivegreen', 'darkslateblue', 'darkslategrey', 'lightcoral']

    rmm = True if len(markers_dict) == 3*len(masses) else False
    total_rmm = gx.plot(0, 0, alpha=0.5, label=f'Total angular momentum', linewidth=3.0, color='DarkSlateGrey')[0]
            

    positions = {name: ax.plot(1, 0, linestyle='', marker='o', alpha=0.6, color=colors[i], markersize=6.0, label=mass)[0] for i, (name, mass) in enumerate(masses.items())}
    paths = {name: ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=2.0)[0] for i, (name, _) in enumerate(masses.items())}
    if rmm:
        # Track positions and paths for the objects and angular momentum markers
        xpositions = {name: ax.plot(1, 0, linestyle='', marker='o', alpha=0.5, color=colors[i], markersize=3.0)[0] for i, (name, _) in enumerate(masses.items())}
        xpaths = {name: ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=1.0)[0] for i, (name, _) in enumerate(masses.items())}
        ypositions = {name: ax.plot(1, 0, linestyle='', marker='o', alpha=0.5, color=colors[i], markersize=3.0)[0] for i, (name, _) in enumerate(masses.items())}
        ypaths = {name: ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=1.0)[0] for i, (name, _) in enumerate(masses.items())}

        internal_rmms = {name: gx.plot(0, 0, alpha=0.75, linewidth=2.0, linestyle=':', color=colors[i])[0] for i, (name, _) in enumerate(masses.items())}
        external_rmms = {name: gx.plot(0, 0, alpha=0.75, linewidth=2.0, linestyle='--', color=colors[i])[0] for i, (name, _) in enumerate(masses.items())}
        rmms = {name: gx.plot(0, 0, alpha=0.5, linewidth=2.0, color=colors[i])[0] for i, (name, _) in enumerate(masses.items())}

    point_line = ax.plot(point[0]*1000, point[1]*1000, linestyle='', marker='x', alpha=0.6, color='black', markersize=3.0, label='Fixed point')[0]

    # Numer of steps to traceback
    traceback_step = int(traceback_time*camera_speed)
    title = ax.set_title(f'{filename}')

    rmm_min = 0
    rmm_max = 0

    internal_rmm = {}
    external_rmm = {}
    # Initialize total angularmomentum, both internal and external
    total_angularmomentum = np.zeros(len(ts0))
    total_external_angularmomentum = np.zeros(len(ts0))

    for name, mass in masses.items():
        # Center of mass
        cm = markers_dict[f"{name}cm"][::sample_stepsize].T * 0.001
        xs, ys, zs = cm
        # Remove z component
        cm = np.array([xs, ys, len(xs)*[0]]).T

        # Rotation point relative center point
        rot = markers_dict[f"{name[0]}x"][::sample_stepsize].T * 0.001 - cm.T
        rxs, rys, rzs = rot
        # Remove z component
        rot = np.array([rxs, rys, len(rxs)*[0]]).T
        rot1 = rot[:-1]
        rot2 = rot[1:]

        # Change in angle
        theta = np.cross(rot1, rot2).T[2]/(np.linalg.norm(rot1, axis=1)*np.linalg.norm(rot2, axis=1))
        
        # Change in velocity
        vs = frequency * (cm[1:] - cm[:-1])

        # Momentum of inertia relative to point
        disc_moi = [moment_of_inertia[f"{name[1:]}"]]*len(vs)
        rs = cm-[point]*len(cm)
        moi =  disc_moi # + mass*np.linalg.norm(rs)**2
        
        # Internal and external angular momentum
        internal_rmm[name] = theta*moi*frequency
        external_rmm[name] = np.cross(rs[1:], mass*vs).T[2]
        
        total_angularmomentum += internal_rmm[name]
        total_external_angularmomentum += external_rmm[name]

        rmm_min = np.min([rmm_min, *internal_rmm[name], *external_rmm[name], *(internal_rmm[name] + external_rmm[name]), *(total_angularmomentum + total_external_angularmomentum)])
        rmm_max = np.max([rmm_max, *internal_rmm[name], *external_rmm[name], *(internal_rmm[name] + external_rmm[name]), *(total_angularmomentum + total_external_angularmomentum)])
        

    # Go through each frame
    def update_frame(num): 
        step = int(stepsize*num + frame_count*start) # Current step, offseted from start
        traceback = max(step-traceback_step, 0) # 'End' of traceback
        title = ax.set_title(f'{filename} | {ts[step]-ts[0]:.3f} s')

        # Loop through all objects and append positions and paths to graphical line
        for name, _ in masses.items():
            cm = markers_dict[f"{name}cm"].T
            positions[name].set_data([cm[0][step]], [cm[1][step]])
            paths[name].set_data([cm[0][traceback:step]], [cm[1][traceback:step]])  # Path is traced
            if rmm:
                # Angular tracks
                rotx = markers_dict[f"{name[0]}x"].T
                roty = markers_dict[f"{name[0]}y"].T
                xpositions[name].set_data([rotx[0][step]], [rotx[1][step]])
                ypositions[name].set_data([roty[0][step]], [roty[1][step]])
                # Trace the angular tracks relative to the objects position
                # Subtract the objects path from the angular track and add objects current position
                xpaths[name].set_data(cm[0][step] + np.array(rotx[0][traceback:step]) - np.array(cm[0][traceback:step]), \
                                      cm[1][step] + np.array(rotx[1][traceback:step]) - np.array(cm[1][traceback:step]))  
                ypaths[name].set_data(cm[0][step] + np.array(roty[0][traceback:step]) - np.array(cm[0][traceback:step]), \
                                      cm[1][step] + np.array(roty[1][traceback:step]) - np.array(cm[1][traceback:step]))        
                internal_rmms[name].set_data(ts0[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize], \
                                            internal_rmm[name][int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize])
                external_rmms[name].set_data(ts0[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize], \
                                             external_rmm[name][int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize])
                rmms[name].set_data(ts0[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize], \
                                    internal_rmm[name][int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize] \
                                        + external_rmm[name][int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize])

        if rmm:
            total_rmm.set_data(ts0[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize], \
                               total_angularmomentum[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize] + total_external_angularmomentum[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize])
            return *positions.values(), *xpositions.values(), *ypositions.values(), *paths.values(), *xpaths.values(), *ypaths.values(), *internal_rmms.values(), *external_rmms.values(), *rmms.values(), total_rmm, point_line, title, 
        else:
            return *positions.values(), *paths.values(), point_line, title

    ax.set_aspect('equal', 'box')
    marginal = 50
    ax.set(xlim=(x_min - marginal, x_max + marginal), xlabel='X (mm)')
    ax.set(ylim=(y_min - marginal, y_max + marginal), ylabel='Y (mm)')
    gx.set(xlim=(ts0[0], ts0[-1]), xlabel='Time (s)')
    gx.set(ylim=(rmm_min, rmm_max))

    fig.legend()
    # Generate animation with set number of frames and interval between frames (in ms)
    ani = animation.FuncAnimation(fig, update_frame, int((end-start) * frame_count/stepsize), interval=1000*stepsize/actual_speed, blit=True)
    return ani