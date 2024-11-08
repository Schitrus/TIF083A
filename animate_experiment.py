import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import numpy as np

from parse_script import parse_marker_data

# Precalculated moment of inertias
moment_of_inertia = {"5.9": 3.084225e-6, "14.6": 4.5625e-6, "27.7": 8.65625e-6, "31.6": 0.00001315025, "34.0": 6.8e-6, \
                     "34.3": 6.83e-6, "22.9": 0.0000152542625, "82.2": 0.000038233275, "82.4": 0.0000383263, \
                        "28.3": 0.0000092010375, "43.5": 0.00008632575, "43.3": 0.0000845703125, "396.1": 0.0003083054}

# Predetermined unique colors
colors = ['cornflowerblue', 'crimson', 'green', 'coral', 'darkorchid', 
          'hotpink', 'lightseagreen', 'sandybrown', 'springgreen', 'violet',
          'tomato', 'forestgreen', 'darkslateblue', 'darkslategrey', 'lightcoral']

total_color = colors[11]

def running_mean(x, y, z, N):
    """ 
    x, y, z: position
    N: window size
    """
    cumsumx = np.cumsum(np.insert(x, 0, 0))
    cumsumy = np.cumsum(np.insert(y, 0, 0))
    cumsumz = np.cumsum(np.insert(z, 0, 0))
    # The length of a smoothed array diminishes with 
    # the window_size, thus one must decrease by the 
    # same amount the length of the time array.
    xs = (cumsumx[N:] - cumsumx[:-N]) / float(N)  # smoothed x array
    ys = (cumsumy[N:] - cumsumy[:-N]) / float(N)  # smoothed x array
    zs = (cumsumz[N:] - cumsumz[:-N]) / float(N)  # smoothed x array
    return xs, ys, zs

class ExperimentGrapher:
    frequency = 500

    def __init__(self, filepath):
        self.filename = filepath.split('/')[-1].split('.')[0]
        self.markers, self.time_points, self.masses = parse_marker_data(filepath)
        self.experiment_type = 3
        self.interpret_data()
        self.smooth_data(50)
        self.construct_data(True)

    def interpret_data(self):
        self.frame_count = len(self.time_points)

        if self.filename.startswith("1D"):
            self.experiment_type = 1
        elif self.filename.startswith("S5") or self.filename.startswith("S6"):
            self.experiment_type = 2
        else:
            self.experiment_type = 3

        self.center_masses = {}
        self.rotational_markers = {}
        self.moment_of_inertias = {}
        self.title_name = self.markers.get("title")
        self.collision_time = self.markers.get('collision')
        for name, _ in self.masses.items():
            self.center_masses[name] = self.markers[f"{name}cm"].T * 0.001
            if self.experiment_type == 3:
                self.rotational_markers[name] = np.array([self.markers[f"{name[0]}x"].T, self.markers[f"{name[0]}y"].T]) * 0.001
                self.moment_of_inertias[name] = moment_of_inertia[f"{name[1:]}"]



    def smooth_data(self, window_size):
        self.smooth_center_masses = {}
        self.smooth_rotational_markers = {}
        self.smooth_time_points = np.array(self.time_points[window_size//2:-window_size//2]) - self.time_points[window_size//2]
        for name, _ in self.masses.items():
            self.smooth_center_masses[name] = running_mean(*self.center_masses[name], window_size)
            if self.experiment_type == 3:
                self.smooth_rotational_markers[name] = running_mean(*self.rotational_markers[name][0], window_size)

    def construct_data(self, ignore_z):
        if self.experiment_type == 3:
            self.reference_point = np.array([0.45,0.45,0])
        self.smooth_rotational_vector = {}
        for name, _ in self.masses.items():
            if ignore_z:
                self.smooth_center_masses[name] = np.insert(self.smooth_center_masses[name][:-1], 2, 0, axis=0)
            if self.experiment_type == 3:
                if ignore_z:
                    self.smooth_rotational_markers[name] = np.insert(self.smooth_rotational_markers[name][:-1], 2, 0, axis=0)
                self.smooth_rotational_vector[name] = self.smooth_rotational_markers[name] - self.smooth_center_masses[name]
    
    def generate(self):
        self.calculate_values()
        self.setup_figure()
        self.generate_animation()

    def setup_figure(self):
        self.fig = plt.figure()
        self.fig.set_dpi(120)
        self.fig.set_layout_engine(layout='constrained')
        self.fig.suptitle(self.title_name, fontsize=16)

        if self.experiment_type == 1:
            self.fig.set_figheight(6)
            self.fig.set_figwidth(8)
            gridspec = self.fig.add_gridspec(ncols=2, nrows=2, width_ratios=[2, 5])

            self.overview_ax = self.fig.add_subplot(gridspec[:,0])
            self.overview_ax.set_aspect('equal')
            self.overview_title = self.overview_ax.set_title("Översikt i xy-planet", fontsize=10)

            self.rm_ax = self.fig.add_subplot(gridspec[0,1])
            self.rm_ax.grid(True, axis='y', linestyle='-.', color='grey')
            self.rm_ax.grid(True, axis='x', linestyle=':', color='darkgrey')
            self.rm_ax.set_xlabel('Tid (s)')
            self.rm_ax.set_ylabel('Rörelsemängd (kgm/s)')
            self.rm_ax.set_title(f'Rörelsemängd före och efter stöt')

            self.energy_ax = self.fig.add_subplot(gridspec[1,1])
            self.energy_ax.grid(True, axis='y', linestyle='-.', color='grey')
            self.energy_ax.grid(True, axis='x', linestyle=':', color='darkgrey')
            self.energy_ax.set_xlabel('Tid (s)')
            self.energy_ax.set_ylabel('Energi (kgm2/s2)')
            self.energy_ax.set_title(f'Kinetisk energi före och efter stöt')
        elif self.experiment_type == 2:
            self.fig.set_figheight(8)
            self.fig.set_figwidth(8)
            gridspec = self.fig.add_gridspec(ncols=1, nrows=2, height_ratios=[2, 1])

            self.overview_ax = self.fig.add_subplot(gridspec[0,0])
            self.overview_ax.set_aspect('equal')
            self.overview_title = self.overview_ax.set_title("Översikt i xy-planet")

            self.rm_ax = self.fig.add_subplot(gridspec[1,0])
            self.rm_ax.grid(True, axis='y', linestyle='-.', color='grey')
            self.rm_ax.grid(True, axis='x', linestyle=':', color='darkgrey')
            self.rm_ax.set_xlabel('Tid (s)')
            self.rm_ax.set_ylabel('Rörelsemängd (kgm/s)')
            self.rm_ax.set_title(f'Rörelsemängd före och efter stöt')
        elif self.experiment_type == 3:
            self.fig.set_figheight(9)
            self.fig.set_figwidth(12)
            gridspec = self.fig.add_gridspec(ncols=2, nrows=2, width_ratios=[2, 3])

            self.overview_ax = self.fig.add_subplot(gridspec[0,0])
            self.overview_ax.set_aspect('equal')
            self.overview_title = self.overview_ax.set_title("Översikt i xy-planet", fontsize=10)
            
            self.energy_ax = self.fig.add_subplot(gridspec[0,1])
            self.energy_ax.grid(True, axis='y', linestyle='-.', color='grey')
            self.energy_ax.grid(True, axis='x', linestyle=':', color='darkgrey')
            self.energy_ax.set_xlabel('Tid (s)')
            self.energy_ax.set_ylabel('Energi (kgm2/s2)')
            self.energy_ax.set_title(f'Kinetisk energi före och efter stöt')

            self.rm_ax = self.fig.add_subplot(gridspec[1,0])
            self.rm_ax.grid(True, axis='y', linestyle='-.', color='grey')
            self.rm_ax.grid(True, axis='x', linestyle=':', color='darkgrey')
            self.rm_ax.set_xlabel('Tid (s)')
            self.rm_ax.set_ylabel('Rörelsemängd (kgm/s)')
            self.rm_ax.set_title(f'Rörelsemängd före och efter stöt')

            self.rmm_ax = self.fig.add_subplot(gridspec[1,1])
            self.rmm_ax.grid(True, axis='y', linestyle='-.', color='grey')
            self.rmm_ax.grid(True, axis='x', linestyle=':', color='darkgrey')
            self.rmm_ax.set_xlabel('Tid (s)')
            self.rmm_ax.set_ylabel('Rörelsemängdsmoment (kgm2/s)')
            self.rmm_ax.set_title(f'Rörelsemängdsmoment före och efter stöt')
        else:
            print("ERROR: Invalid experiment type")

        self.overview_ax.set_aspect('equal')
        self.overview_ax.grid(True, linestyle=':', color='darkgrey')
        self.overview_ax.set_xlabel('X (m)')
        self.overview_ax.set_ylabel('Y (m)')

    
    def calculate_1Drm_values(self):
        self.rms = {}
        self.rm_min, self.rm_max = 0, 0
        for name, mass in self.masses.items():
            cm = self.smooth_center_masses[name].T
            # Change in velocity
            vs = self.frequency * (cm[1:] - cm[:-1])
            # Momentum along velocity direction, positive if y > 0 otherwise negative
            rm = np.where(np.greater_equal(vs.T[1], 0), 1, -1) * np.linalg.norm(vs, axis=1)*mass
            self.rms[name] = rm
            self.rm_min = np.min([self.rm_min, *rm])
            self.rm_max = np.max([self.rm_max, *rm])
        self.total_rm = np.sum([*self.rms.values()], axis=0) 
        self.rm_min = np.min([self.rm_min, *self.total_rm])
        self.rm_max = np.max([self.rm_max, *self.total_rm])

    def calculate_1Denergy_values(self):
        self.energies = {}
        self.energy_min, self.energy_max = 0, 0
        for name, mass in self.masses.items():
            cm = self.smooth_center_masses[name].T
            # Change in velocity
            vs = self.frequency * (cm[1:] - cm[:-1])
            # Kinetic energy
            energy = 0.5*mass*np.linalg.norm(vs, axis=1)**2
            self.energies[name] = energy
            self.energy_min = np.min([self.energy_min, *energy])
            self.energy_max = np.max([self.energy_max, *energy])
        self.total_energy = np.sum([*self.energies.values()], axis=0) 
        self.energy_min = np.min([self.energy_min, *self.total_energy])
        self.energy_max = np.max([self.energy_max, *self.total_energy])

    def calculate_2Drm_values(self):
        self.rmxs = {}
        self.rmys = {}
        self.rm_min, self.rm_max = 0, 0
        for name, mass in self.masses.items():
            cm = self.smooth_center_masses[name].T

            # Change in velocity
            vs = self.frequency * (cm[1:] - cm[:-1])

            # Internal and external angular momentum
            rmx = vs.T[0]*mass
            rmy = vs.T[1]*mass
            self.rmxs[name] = rmx
            self.rmys[name] = rmy

            self.rm_min = np.min([self.rm_min, *rmx, *rmy])
            self.rm_max = np.max([self.rm_max, *rmx, *rmy])
        self.total_rmx = np.sum([*self.rmxs.values()], axis=0)
        self.total_rmy = np.sum([*self.rmys.values()], axis=0)

        self.rm_min = np.min([self.rm_min, *self.total_rmx, *self.total_rmy])
        self.rm_max = np.max([self.rm_max, *self.total_rmx, *self.total_rmy])

    def calculate_2Denergy_values(self):
        self.internal_energies = {}
        self.external_energies = {}
        self.energies = {}
        self.energy_min, self.energy_max = 0, 0
        for name, mass in self.masses.items():
            cm = self.smooth_center_masses[name].T
            rot = self.smooth_rotational_vector[name].T
            rot2, rot1 = rot[1:], rot[:-1]

            # Change in angle
            theta = np.cross(rot1, rot2).T[2]/(np.linalg.norm(rot1, axis=1)*np.linalg.norm(rot2, axis=1))

            # Change in velocity
            vs = self.frequency * (cm[1:] - cm[:-1])

            # Momentum of inertia
            moi = self.moment_of_inertias[name]

            # Internal and external energy
            internal_energy = 0.5*moi*(theta*self.frequency)**2
            external_energy = 0.5*mass*np.linalg.norm(vs, axis=1)**2

            energy = internal_energy + external_energy
            self.internal_energies[name] = internal_energy
            self.external_energies[name] = external_energy
            self.energies[name] = energy

            self.energy_min = np.min([self.energy_min, *internal_energy, *external_energy, *energy])
            self.energy_max = np.max([self.energy_max, *internal_energy, *external_energy, *energy])
        self.total_energy = np.sum([*self.energies.values()], axis=0)

        self.energy_min = np.min([self.energy_min, *self.total_energy])
        self.energy_max = np.max([self.energy_max, *self.total_energy])

    def calculate_rmm_values(self):
        self.internal_rmms = {}
        self.external_rmms = {}
        self.rmms = {}
        self.rmm_min, self.rmm_max = 0, 0
        for name, mass in self.masses.items():
            cm = self.smooth_center_masses[name].T
            rot = self.smooth_rotational_vector[name].T
            rot2, rot1 = rot[1:], rot[:-1]

            # Change in angle
            theta = np.cross(rot1, rot2).T[2]/(np.linalg.norm(rot1, axis=1)*np.linalg.norm(rot2, axis=1))

            # Change in velocity
            vs = self.frequency * (cm[1:] - cm[:-1])

            # Momentum of inertia relative to point
            rs = cm-self.reference_point
            moi = self.moment_of_inertias[name]

            # Internal and external angular momentum
            internal_rmm = theta*moi*self.frequency
            external_rmm = np.cross(rs[1:], mass*vs).T[2]
            rmm = internal_rmm + external_rmm
            self.internal_rmms[name] = internal_rmm
            self.external_rmms[name] = external_rmm
            self.rmms[name] = rmm

            self.rmm_min = np.min([self.rmm_min, *internal_rmm, *external_rmm, *rmm])
            self.rmm_max = np.max([self.rmm_max, *internal_rmm, *external_rmm, *rmm])
        self.total_rmm = np.sum([*self.rmms.values()], axis=0)

        self.rmm_min = np.min([self.rmm_min, *self.total_rmm])
        self.rmm_max = np.max([self.rmm_max, *self.total_rmm])


    def calculate_values(self):
        if self.experiment_type == 1:
            self.calculate_1Drm_values()
            self.calculate_1Denergy_values()
        elif self.experiment_type == 2:
            self.calculate_2Drm_values()
        elif self.experiment_type == 3:
            self.calculate_2Drm_values()
            self.calculate_2Denergy_values()
            self.calculate_rmm_values()
        else:
            print("ERROR: Invalid experiment type")
        
        if self.experiment_type == 1:
            self.x_min = np.min([np.min(cm[0]) for cm in self.center_masses.values()])
            self.x_max = np.max([np.max(cm[0]) for cm in self.center_masses.values()])
            self.y_min = np.min([np.min(cm[1]) for cm in self.center_masses.values()])
            self.y_max = np.max([np.max(cm[1]) for cm in self.center_masses.values()])
        elif self.experiment_type == 2:
            self.x_min = np.min([np.min([cm[0], cm[1]]) for cm in self.center_masses.values()])
            self.y_min = self.x_min
            self.x_max = np.max([np.max([cm[0], cm[1]]) for cm in self.center_masses.values()])
            self.y_max = self.x_max
        else:
            self.x_min = np.min([np.min([cm[0], rotx[0], roty[0]]) for cm, rotx, roty in zip(self.center_masses.values(), *self.rotational_markers.values())])
            self.x_max = np.max([np.max([cm[0], rotx[0], roty[0]]) for cm, rotx, roty in zip(self.center_masses.values(), *self.rotational_markers.values())])
            self.y_min = np.min([np.min([cm[1], rotx[1], roty[1]]) for cm, rotx, roty in zip(self.center_masses.values(), *self.rotational_markers.values())])
            self.y_max = np.max([np.max([cm[1], rotx[1], roty[1]]) for cm, rotx, roty in zip(self.center_masses.values(), *self.rotational_markers.values())])


    def generate_animation(self, speed=1.0, traceback_time=1.0, start=0.0, end=1.0):
        self.setup_animation(speed, traceback_time, start, end)

        # Camera speed is 500Hz
        camera_speed = 500
        actual_speed = camera_speed * speed

        # Goal to have 50 fps in animation, more is buggy
        fps = 50
        stepsize=max(actual_speed//fps, 1) # How many steps to go for each frame

        # Numer of steps to traceback
        traceback_step = int(traceback_time*camera_speed)

        marginal = 0.050
        self.overview_ax.set_xlim(left=self.x_min - marginal, right=self.x_max + marginal)
        self.overview_ax.set_ylim(bottom=self.y_min - marginal, top=self.y_max + marginal)
        self.rm_ax.set_xlim(left=self.smooth_time_points[0], right=self.smooth_time_points[-1])
        self.rm_ax.set_ylim(bottom=self.rm_min - 0.05*(self.rm_max-self.rm_min), top=self.rm_max + 0.05*(self.rm_max-self.rm_min))
        if self.experiment_type != 2:
            self.energy_ax.set_xlim(left=self.smooth_time_points[0], right=self.smooth_time_points[-1])
            self.energy_ax.set_ylim(bottom=self.energy_min - 0.05*(self.energy_max-self.energy_min), top=self.energy_max + 0.05*(self.energy_max-self.energy_min))
        if self.experiment_type == 3:
            self.rmm_ax.set_xlim(left=self.smooth_time_points[0], right=self.smooth_time_points[-1])
            self.rmm_ax.set_ylim(bottom=self.rmm_min - 0.05*(self.rmm_max-self.rmm_min), top=self.rmm_max + 0.05*(self.rmm_max-self.rmm_min))

        # Go through each frame
        def update_frame(num): 
            start_step = int(self.frame_count*start)
            current_step = int(stepsize*num) + start_step # Current step, offseted from start
            traceback = max(current_step-traceback_step, 0) # 'End' of traceback

            return *self.graph_animation(start_step, current_step, traceback), 

        self.overview_ax.legend(fontsize=6)
        self.rm_ax.legend(fontsize=6)
        if self.experiment_type != 2:
            self.energy_ax.legend(fontsize=6)
        if self.experiment_type == 3:
            self.rmm_ax.legend(fontsize=6)
        # Generate animation with set number of frames and interval between frames (in ms)
        self.experiment_animation = animation.FuncAnimation(self.fig, update_frame, int((end-start) * self.frame_count/stepsize), interval=1000*stepsize/actual_speed, blit=True)

    def setup_animation(self, speed, traceback_time, start, end):            
        self.position_points = {name: self.overview_ax.plot(1, 0, linestyle='', marker='o', alpha=0.6, color=colors[i], markersize=6.0, label=f'{1000*mass:.1f}g')[0] for i, (name, mass) in enumerate(self.masses.items())}
        self.path_lines = {name: self.overview_ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=2.0)[0] for i, (name, _) in enumerate(self.masses.items())}
        
        if self.experiment_type == 1:
            self.rm_lines = {name: self.rm_ax.plot(0, 0, alpha=0.6, linewidth=2.0, color=colors[i], label=f'{1000*mass:.1f}g - rörelsemängd')[0] for i, (name, mass) in enumerate(self.masses.items())}
            self.total_rm_line = self.rm_ax.plot(0, 0, alpha=0.75, label=f'Total rörelsemängd', linewidth=3.0, color=total_color)[0]
        else:
            self.rmx_lines = {name: self.rm_ax.plot(0, 0, alpha=0.6, linewidth=2.0, linestyle=':', color=colors[i], label=f'{1000*mass:.1f}g - rörelsemängd x-axel')[0] for i, (name, mass) in enumerate(self.masses.items())}
            self.rmy_lines = {name: self.rm_ax.plot(0, 0, alpha=0.6, linewidth=2.0, linestyle='--', color=colors[i], label=f'{1000*mass:.1f}g - rörelsemängd y-axel')[0] for i, (name, mass) in enumerate(self.masses.items())}
            self.total_rmx_line = self.rm_ax.plot(0, 0, alpha=0.75, label=f'Total rörelsemängd x-axel', linestyle=':', linewidth=3.0, color=total_color)[0]
            self.total_rmy_line = self.rm_ax.plot(0, 0, alpha=0.75, label=f'Total rörelsemängd y-axel', linestyle='--', linewidth=3.0, color=total_color)[0]

        if self.experiment_type != 2:
            self.energy_lines = {name: self.energy_ax.plot(0, 0, alpha=0.6, linewidth=2.0, color=colors[i], label=f'{1000*mass:.1f}g - total energi')[0] for i, (name, mass) in enumerate(self.masses.items())}
            self.total_energy_line = self.energy_ax.plot(0, 0, alpha=0.75, label=f'Total rörelseenergi', linewidth=3.0, color=total_color)[0]
        if self.experiment_type == 3:
            # Track positions and paths for the objects and angular momentum markers
            self.xrotational_points = {name: self.overview_ax.plot(1, 0, linestyle='', marker='o', alpha=0.5, color=colors[i], markersize=3.0)[0] for i, (name, _) in enumerate(self.masses.items())}
            self.xrotationalpath_lines = {name: self.overview_ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=1.0)[0] for i, (name, _) in enumerate(self.masses.items())}
            self.yrotational_points = {name: self.overview_ax.plot(1, 0, linestyle='', marker='o', alpha=0.5, color=colors[i], markersize=3.0)[0] for i, (name, _) in enumerate(self.masses.items())}
            self.yrotationalpath_lines = {name: self.overview_ax.plot(1, 0, linestyle='-', alpha=0.5, color=colors[i], linewidth=1.0)[0] for i, (name, _) in enumerate(self.masses.items())}

            self.internal_energy_lines = {name: self.energy_ax.plot(0, 0, alpha=0.6, linewidth=2.0, linestyle=':', color=colors[i], label=f'{1000*mass:.1f}g - inre energi')[0] for i, (name, mass) in enumerate(self.masses.items())}
            self.external_energy_lines = {name: self.energy_ax.plot(0, 0, alpha=0.6, linewidth=2.0, linestyle='--', color=colors[i], label=f'{1000*mass:.1f}g - yttre energi')[0] for i, (name, mass) in enumerate(self.masses.items())}

            self.internal_rmm_lines = {name: self.rmm_ax.plot(0, 0, alpha=0.6, linewidth=2.0, linestyle=':', color=colors[i], label=f'{1000*mass:.1f}g - intre RMM')[0] for i, (name, mass) in enumerate(self.masses.items())}
            self.external_rmm_lines = {name: self.rmm_ax.plot(0, 0, alpha=0.6, linewidth=2.0, linestyle='--', color=colors[i], label=f'{1000*mass:.1f}g - yttre RMM')[0] for i, (name, mass) in enumerate(self.masses.items())}
            self.rmm_lines = {name: self.rmm_ax.plot(0, 0, alpha=0.75, linewidth=2.0, color=colors[i])[0] for i, (name, _) in enumerate(self.masses.items())}
            self.total_rmm_line = self.rmm_ax.plot(0, 0, alpha=0.75, label=f'Total rörelsemängdsmoment', linewidth=3.0, color=total_color)[0]

            self.point_line = self.overview_ax.plot(self.reference_point[0], self.reference_point[1], linestyle='', marker='x', alpha=0.6, color='black', markersize=4.0, label='Referenspunkt')[0]

    def graph_overview_rotation(self, step, traceback):
        draw_pile = []
        for name, _ in self.masses.items():
            # Angular tracks
            cm = self.center_masses[name]
            rotx = self.rotational_markers[name][0]
            roty = self.rotational_markers[name][1]
            self.xrotational_points[name].set_data([rotx[0][step]], [rotx[1][step]])
            self.yrotational_points[name].set_data([roty[0][step]], [roty[1][step]])
            # Trace the angular tracks relative to the objects position
            # Subtract the objects path from the angular track and add objects current position
            self.xrotationalpath_lines[name].set_data(cm[0][step] + np.array(rotx[0][traceback:step]) - np.array(cm[0][traceback:step]), \
                                  cm[1][step] + np.array(rotx[1][traceback:step]) - np.array(cm[1][traceback:step]))  
            self.yrotationalpath_lines[name].set_data(cm[0][step] + np.array(roty[0][traceback:step]) - np.array(cm[0][traceback:step]), \
                                  cm[1][step] + np.array(roty[1][traceback:step]) - np.array(cm[1][traceback:step])) 

            draw_pile.extend([self.xrotational_points[name], self.yrotational_points[name], self.xrotationalpath_lines[name], self.yrotationalpath_lines[name]])
        return *draw_pile,

    def graph_overview(self, step, traceback):
        draw_pile = []
        for name, _ in self.masses.items():
            cm = self.center_masses[name]
            self.position_points[name].set_data([cm[0][step]], [cm[1][step]])
            self.path_lines[name].set_data([cm[0][traceback:step]], [cm[1][traceback:step]])  # Path is traced
            draw_pile.append(self.position_points[name])
        if self.experiment_type == 3:
            draw_pile.extend(self.graph_overview_rotation(step, traceback))
        return *draw_pile,
    
    def graph_plot_rmm(self, begin, stop):
        draw_pile = []
        for name, _ in self.masses.items():
            self.internal_rmm_lines[name].set_data(self.smooth_time_points[begin:stop], self.internal_rmms[name][begin:stop])
            self.external_rmm_lines[name].set_data(self.smooth_time_points[begin:stop], self.external_rmms[name][begin:stop])
            self.rmm_lines[name].set_data(self.smooth_time_points[begin:stop], self.rmms[name][begin:stop])
            draw_pile.extend([self.internal_rmm_lines[name], self.external_rmm_lines[name], self.rmm_lines[name]])
        self.total_rmm_line.set_data(self.smooth_time_points[begin:stop], self.total_rmm[begin:stop])
        draw_pile.append(self.total_rmm_line)
        return *draw_pile,

    def graph_plot_rm(self, begin, stop):
        draw_pile = []
        for name, _ in self.masses.items():
            if self.experiment_type == 1:
                self.rm_lines[name].set_data(self.smooth_time_points[begin:stop], self.rms[name][begin:stop])
                draw_pile.append(self.rm_lines[name])
            else:
                self.rmx_lines[name].set_data(self.smooth_time_points[begin:stop], self.rmxs[name][begin:stop])
                self.rmy_lines[name].set_data(self.smooth_time_points[begin:stop], self.rmys[name][begin:stop])
                draw_pile.extend([self.rmx_lines[name], self.rmy_lines[name]])

        if self.experiment_type == 1:
            self.total_rm_line.set_data(self.smooth_time_points[begin:stop], self.total_rm[begin:stop])
            draw_pile.append(self.total_rm_line)
        else:
            self.total_rmx_line.set_data(self.smooth_time_points[begin:stop], self.total_rmx[begin:stop])
            self.total_rmy_line.set_data(self.smooth_time_points[begin:stop], self.total_rmy[begin:stop])
            draw_pile.extend([self.total_rmx_line, self.total_rmy_line])

        return *draw_pile,

    def graph_plot_energy(self, begin, stop):
        draw_pile = []
        for name, _ in self.masses.items():
            if self.experiment_type == 3:
                self.internal_energy_lines[name].set_data(self.smooth_time_points[begin:stop], self.internal_energies[name][begin:stop])
                self.external_energy_lines[name].set_data(self.smooth_time_points[begin:stop], self.external_energies[name][begin:stop])
                draw_pile.extend([self.internal_energy_lines[name], self.external_energy_lines[name]])
            self.energy_lines[name].set_data(self.smooth_time_points[begin:stop], self.energies[name][begin:stop])
            draw_pile.append(self.energy_lines[name])
        self.total_energy_line.set_data(self.smooth_time_points[begin:stop], self.total_energy[begin:stop])
        draw_pile.append(self.total_energy_line)
        return *draw_pile,

    def graph_plot(self, start_step, current_step):
        draw_pile = [] 
        draw_pile.extend(self.graph_plot_rm(start_step, current_step))
        if self.experiment_type != 2:
            draw_pile.extend(self.graph_plot_energy(start_step, current_step))
        if self.experiment_type == 3:
            draw_pile.extend(self.graph_plot_rmm(start_step, current_step))
        return *draw_pile,

    def graph_animation(self, start_step, current_step, traceback):
        self.overview_title.set_text(f'Översikt i xy-planet | {self.time_points[current_step]-self.time_points[0]:.3f} s')
        return *self.graph_overview(current_step, traceback), *self.graph_plot(start_step, current_step), self.overview_title,

    def save_animation(self, output):
        self.experiment_animation.save(f'{output}/{self.filename}.gif')