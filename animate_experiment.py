import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from data_parser import read_data
from parse_script import parse_marker_data

# Precalculated moment of inertias
moment_of_inertia = {"5.9": 3.084225e-6, "14.6": 4.5625e-6, "27.7": 8.65625e-6, "31.6": 0.00001315025, "34.0": 6.8e-6, \
                     "34.3": 6.83e-6, "22.9": 0.0000152542625, "82.2": 0.000038233275, "82.4": 0.0000383263, \
                        "28.3": 0.0000092010375, "43.5": 0.00008632575, "43.3": 0.0000845703125, "396.1": 0.0003083054}

def generate_animation(filepath, rmm=False, speed=1.0, traceback_time=1.0, start=0.0, end=1.0):
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
    obj_names, ts, objs = read_data(filepath)
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
    x_min = np.min([np.min(obj[0]) for obj in objs])
    x_max = np.max([np.max(obj[0]) for obj in objs])
    y_min = np.min([np.min(obj[1]) for obj in objs])
    y_max = np.max([np.max(obj[1]) for obj in objs])

    # Predetermined unique colors
    colors = ['cornflowerblue', 'crimson', 'green', 'coral', 'darkorchid', 
              'hotpink', 'lightseagreen', 'sandybrown', 'springgreen', 'violet',
              'tomato', 'darkolivegreen', 'darkslateblue', 'darkslategrey', 'lightcoral']
    
    positions = []
    paths = []
    xpositions = []
    ypositions = []
    xpaths = []
    ypaths = []

    internal_rmms = []
    external_rmms = []
    rmms = []
    total_rmm = gx.plot(0, 0, alpha=0.5, label=f'Total angular momentum', linewidth=3.0, color='DarkSlateGrey')[0]
            

    if rmm:
        # Track positions and paths for the objects and angular momentum markers
        positions = [ax.plot(1, 0, linestyle='', marker='o', alpha=0.6, color=colors[i], markersize=6.0, label=obj_names[i*3])[0] for i in range(len(objs)//3)]
        paths = [ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=2.0)[0] for i in range(len(objs)//3)]
        xpositions = [ax.plot(1, 0, linestyle='', marker='o', alpha=0.5, color=colors[i], markersize=3.0)[0] for i in range(len(objs)//3)]
        xpaths = [ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=1.0)[0] for i in range(len(objs)//3)]
        ypositions = [ax.plot(1, 0, linestyle='', marker='o', alpha=0.5, color=colors[i], markersize=3.0)[0] for i in range(len(objs)//3)]
        ypaths = [ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=1.0)[0] for i in range(len(objs)//3)]

        internal_rmms = [gx.plot(0, 0, alpha=0.75, linewidth=2.0, linestyle=':', color=colors[i])[0] for i in range(len(objs)//3)]
        external_rmms = [gx.plot(0, 0, alpha=0.75, linewidth=2.0, linestyle='--', color=colors[i])[0] for i in range(len(objs)//3)]
        rmms = [gx.plot(0, 0, alpha=0.5, linewidth=2.0, color=colors[i])[0] for i in range(len(objs)//3)]
    else:
        positions = [ax.plot(1, 0, linestyle='', marker='o', alpha=0.6, color=colors[i], markersize=6.0, label=obj_names[i])[0] for i in range(len(objs))]
        paths = [ax.plot(1, 0, linestyle='-', alpha=0.6, color=colors[i], linewidth=2.0)[0] for i in range(len(objs))]

    point_line = ax.plot(point[0]*1000, point[1]*1000, linestyle='', marker='x', alpha=0.6, color='black', markersize=3.0, label='Fixed point')[0]

    # Numer of steps to traceback
    traceback_step = int(traceback_time*camera_speed)
    title = ax.set_title(f'{filename}')

    rmm_min = 0
    rmm_max = 0

    internal_rmm = []
    external_rmm = []
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
        internal_rmm.append(theta*moi*frequency)
        external_rmm.append(np.cross(rs[1:], mass*vs).T[2])
        
        total_angularmomentum += internal_rmm[-1]
        total_external_angularmomentum += external_rmm[-1]

        rmm_min = np.min([rmm_min, *internal_rmm[-1], *external_rmm[-1], *(internal_rmm[-1] + external_rmm[-1]), *(total_angularmomentum + total_external_angularmomentum)])
        rmm_max = np.max([rmm_max, *internal_rmm[-1], *external_rmm[-1], *(internal_rmm[-1] + external_rmm[-1]), *(total_angularmomentum + total_external_angularmomentum)])
        

    # Go through each frame
    def update_frame(num): 
        step = int(stepsize*num + frame_count*start) # Current step, offseted from start
        traceback = max(step-traceback_step, 0) # 'End' of traceback
        title = ax.set_title(f'{filename} | {ts[step]-ts[0]:.3f} s')

        # Loop through all objects and append positions and paths to graphical line
        if not rmm:
            for obj, pos, path in zip(objs, positions, paths):
                pos.set_data([obj[0][step]], [obj[1][step]])
                path.set_data(obj[0][traceback:step], obj[1][traceback:step]) # Path is traced
            return *positions, *paths, point_line, title,
        else:
            # Objects and angular tracks
            for obj, xobj, yobj, pos, xpos, ypos, path, xpath, ypath in zip(objs[::3], objs[1::3], objs[2::3], positions, xpositions, ypositions, paths, xpaths, ypaths):
                pos.set_data([obj[0][step]], [obj[1][step]])
                path.set_data(obj[0][traceback:step], obj[1][traceback:step])
                xpos.set_data([xobj[0][step]], [xobj[1][step]])
                ypos.set_data([yobj[0][step]], [yobj[1][step]])
                # Trace the angular tracks relative to the objects position
                # Subtract the objects path from the angular track and add objects current position
                xpath.set_data(obj[0][step] + np.array(xobj[0][traceback:step]) - np.array(obj[0][traceback:step]), obj[1][step] + np.array(xobj[1][traceback:step]) - np.array(obj[1][traceback:step]))
                ypath.set_data(obj[0][step] + np.array(yobj[0][traceback:step]) - np.array(obj[0][traceback:step]), obj[1][step] + np.array(yobj[1][traceback:step]) - np.array(obj[1][traceback:step]))
            
            for i_rmm, e_rmm, i_rmms, e_rmms, sum_rmms in zip(internal_rmm, external_rmm, internal_rmms, external_rmms, rmms):
                i_rmms.set_data(ts0[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize], i_rmm[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize])
                e_rmms.set_data(ts0[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize], e_rmm[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize])
                sum_rmms.set_data(ts0[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize], i_rmm[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize] + e_rmm[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize])
                
            total_rmm.set_data(ts0[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize], total_angularmomentum[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize] + total_external_angularmomentum[int(frame_count*start)//sample_stepsize:int(stepsize*num + frame_count*start)//sample_stepsize])

            return *positions, *xpositions, *ypositions, *paths, *xpaths, *ypaths, *internal_rmms, *external_rmms, *rmms, total_rmm, point_line, title, 

    fig.legend()

    ax.set_aspect('equal', 'box')
    marginal = 50
    ax.set(xlim=(x_min - marginal, x_max + marginal), xlabel='X (mm)')
    ax.set(ylim=(y_min - marginal, y_max + marginal), ylabel='Y (mm)')
    gx.set(xlim=(ts0[0], ts0[-1]), xlabel='Time (s)')
    gx.set(ylim=(rmm_min, rmm_max))

    # Generate animation with set number of frames and interval between frames (in ms)
    ani = animation.FuncAnimation(fig, update_frame, int((end-start) * frame_count/stepsize), interval=1000*stepsize/actual_speed, blit=True)
    return ani