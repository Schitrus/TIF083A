import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from data_parser import read_data


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
    ax = fig.add_subplot()

    # Determines the scale of the figure
    x_min = np.min([np.min(obj[0]) for obj in objs])
    x_max = np.max([np.max(obj[0]) for obj in objs])
    y_min = np.min([np.min(obj[1]) for obj in objs])
    y_max = np.max([np.max(obj[1]) for obj in objs])

    # Predetermined unique colors
    colors = ['cornflowerblue', 'crimson', 'green', 'coral', 'darkorchid', 
              'hotpink', 'lightseagreen', 'sandybrown', 'springgreen', 'violet',
              'tomato', 'darkolivegreen', 'darkslateblue', 'darkslategrey', 'lightcoral']
    
    if rmm:
        # Track positions and paths for the objects and angular momentum markers
        positions = [ax.plot(1, 0, linestyle='', marker='o', alpha=0.6, color=colors[i], markersize=6.0, label=obj_names[i*3])[0] for i in range(len(objs)//3)]
        paths = [ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=2.0)[0] for i in range(len(objs)//3)]
        xpositions = [ax.plot(1, 0, linestyle='', marker='o', alpha=0.5, color=colors[i], markersize=3.0)[0] for i in range(len(objs)//3)]
        xpaths = [ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=1.0)[0] for i in range(len(objs)//3)]
        ypositions = [ax.plot(1, 0, linestyle='', marker='o', alpha=0.5, color=colors[i], markersize=3.0)[0] for i in range(len(objs)//3)]
        ypaths = [ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=1.0)[0] for i in range(len(objs)//3)]
    else:
        positions = [ax.plot(1, 0, linestyle='', marker='o', alpha=0.6, color=colors[i], markersize=6.0, label=obj_names[i])[0] for i in range(len(objs))]
        paths = [ax.plot(1, 0, linestyle='-', alpha=0.6, color=colors[i], linewidth=2.0)[0] for i in range(len(objs))]

    # Numer of steps to traceback
    traceback_step = int(traceback_time/camera_speed)
    title = ax.set_title(f'{filename}')

    # Go through each frame
    def update_frame(num): 
        step = int(stepsize*num + frame_count*start) # Current step, offseted from start
        traceback = max(step-traceback_step, 0) # 'End' of traceback
        title = ax.set_title(f'{filename} | {ts[step]-ts[0]:.3f} s')

        # Loop through all objects and append positions and paths to graphical line
        if not rmm:
            for obj, pos, path in zip(objs, positions, paths):
                pos.set_data(obj[0][step], obj[1][step])
                path.set_data(obj[0][traceback:step], obj[1][traceback:step]) # Path is traced
            return *positions, *paths, title
        else:
            # Objects and angular tracks
            for obj, xobj, yobj, pos, xpos, ypos, path, xpath, ypath in zip(objs[::3], objs[1::3], objs[2::3], positions, xpositions, ypositions, paths, xpaths, ypaths):
                pos.set_data(obj[0][step], obj[1][step])
                path.set_data(obj[0][traceback:step], obj[1][traceback:step])
                xpos.set_data(xobj[0][step], xobj[1][step])
                ypos.set_data(yobj[0][step], yobj[1][step])
                # Trace the angular tracks relative to the objects position
                # Subtract the objects path from the angular track and add objects current position
                xpath.set_data(obj[0][step] + np.array(xobj[0][traceback:step]) - np.array(obj[0][traceback:step]), obj[1][step] + np.array(xobj[1][traceback:step]) - np.array(obj[1][traceback:step]))
                ypath.set_data(obj[0][step] + np.array(yobj[0][traceback:step]) - np.array(obj[0][traceback:step]), obj[1][step] + np.array(yobj[1][traceback:step]) - np.array(obj[1][traceback:step]))
            return *positions, *xpositions, *ypositions, *paths, *xpaths, *ypaths, title

    fig.legend()

    ax.set_aspect('equal', 'box')
    marginal = 50
    ax.set(xlim=(x_min - marginal, x_max + marginal), xlabel='X (mm)')
    ax.set(ylim=(y_min - marginal, y_max + marginal), ylabel='Y (mm)')

    # Generate animation with set number of frames and interval between frames (in ms)
    ani = animation.FuncAnimation(fig, update_frame, int((end-start) * frame_count/stepsize), interval=1000*stepsize/actual_speed, blit=True)
    return ani