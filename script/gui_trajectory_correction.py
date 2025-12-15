# Copyright 2025 Distance Technologies Oy. For internal use only.
#
"""GUI application for trajectory correction parameter tuning."""
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osm_align.core.odometry_correction import OdomCorrector
from osm_align.utils import utils
from osm_align.utils.kitti_utils import angle_dict, cordinta_dict
import pykitti
from scipy.spatial.transform import Rotation
import lanelet2


class TrajectoryCorrectionGUI:
    """GUI application for tuning trajectory correction parameters."""

    def __init__(self, root, basalt_pose_dir, kitti_base_dir):
        """
        Initialize the GUI application.

        Parameters
        ----------
        root : tk.Tk
            Root tkinter window.
        basalt_pose_dir : str
            Base directory for basalt pose files.
        kitti_base_dir : str
            Base directory for KITTI dataset.
        """
        self.root = root
        self.root.title("Trajectory Correction Parameter Tuning")
        self.root.geometry("1800x1000")

        self.basalt_pose_dir = basalt_pose_dir
        self.kitti_base_dir = kitti_base_dir

        # Default parameters
        self.default_args = {
            'pose_segment_size': 50,
            'knn_neighbors': 10,
            'valid_correspondence_threshold': 0.1,
            'icp_error_threshold': 1.0,
            'trimming_ratio': 0.1,
            'min_distance_threshold': 5.0,
            'max_error_consecutive': 1000,
        }

        # Current data
        self.current_seq = '01'
        self.gt_poses = None
        self.basalt_poses = None
        self.points_lane_map = None
        self.lanelet_map = None
        self.poses_corrected = None

        # Plot artists for visibility control
        self.plot_artists = {
            'map': [],
            'gt': None,
            'basalt': None,
            'corrected': None,
        }
        
        # Status text on plot
        self.status_text = None

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Right panel for plot
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Paths display
        paths_frame = ttk.LabelFrame(left_panel, text="Paths", padding="5")
        paths_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(paths_frame, text="Basalt Dir:").pack(anchor=tk.W)
        ttk.Label(paths_frame, text=self.basalt_pose_dir, foreground="gray").pack(anchor=tk.W, padx=(10, 0))

        ttk.Label(paths_frame, text="KITTI Dir:").pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(paths_frame, text=self.kitti_base_dir, foreground="gray").pack(anchor=tk.W, padx=(10, 0))

        # Sequence selection
        seq_frame = ttk.LabelFrame(left_panel, text="Sequence", padding="5")
        seq_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(seq_frame, text="Sequence:").pack(anchor=tk.W)
        self.seq_var = tk.StringVar(value='01')
        # Only include sequences that exist in angle_dict
        available_seqs = [f"{i:02d}" for i in range(11) if f"{i:02d}" in angle_dict]
        seq_combo = ttk.Combobox(seq_frame, textvariable=self.seq_var, 
                                 values=available_seqs, 
                                 state="readonly", width=10)
        seq_combo.pack(anchor=tk.W, padx=(10, 0), pady=(5, 0))
        seq_combo.bind('<<ComboboxSelected>>', self.on_sequence_change)

        # Parameters frame with sliders
        params_frame = ttk.LabelFrame(left_panel, text="Parameters", padding="10")
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.param_vars = {}
        self.param_sliders = {}

        # Parameter ranges and types
        param_config = {
            'pose_segment_size': {'min': 10, 'max': 500, 'step': 10, 'type': int},
            'knn_neighbors': {'min': 0, 'max': 200, 'step': 5, 'type': int},
            'valid_correspondence_threshold': {'min': 0.0, 'max': 1.0, 'step': 0.1, 'type': float},
            'icp_error_threshold': {'min': 0.1, 'max': 10.0, 'step': 0.5, 'type': float},
            'trimming_ratio': {'min': 0.0, 'max': 0.9, 'step': 0.1, 'type': float},
            'min_distance_threshold': {'min': 1.0, 'max': 100.0, 'step': 2.0, 'type': float},
            'max_error_consecutive': {'min': 1, 'max': 100, 'step': 10, 'type': int},
        }

        for param_name, default_value in self.default_args.items():
            config = param_config[param_name]
            var = tk.DoubleVar(value=float(default_value))
            self.param_vars[param_name] = var

            # Label
            label_frame = ttk.Frame(params_frame)
            label_frame.pack(fill=tk.X, pady=(5, 2))
            param_label = ttk.Label(label_frame, text=param_name.replace('_', ' ').title() + ":", font=('TkDefaultFont', 10))
            param_label.pack(side=tk.LEFT)
            value_label = ttk.Label(label_frame, textvariable=var, width=12, font=('TkDefaultFont', 10, 'bold'))
            value_label.pack(side=tk.RIGHT)

            # Slider
            slider = ttk.Scale(
                params_frame,
                from_=config['min'],
                to=config['max'],
                variable=var,
                orient=tk.HORIZONTAL,
                length=400,
                command=lambda v, p=param_name, t=config['type']: self.update_param_value(p, t, v)
            )
            slider.pack(fill=tk.X, pady=(0, 10), ipady=5)
            self.param_sliders[param_name] = slider

        # Visibility controls
        visibility_frame = ttk.LabelFrame(left_panel, text="Plot Visibility", padding="10")
        visibility_frame.pack(fill=tk.X, pady=(0, 10))

        self.visibility_vars = {
            'map': tk.BooleanVar(value=True),
            'gt': tk.BooleanVar(value=True),
            'basalt': tk.BooleanVar(value=True),
            'corrected': tk.BooleanVar(value=True),
        }

        ttk.Checkbutton(visibility_frame, text="Map", variable=self.visibility_vars['map'],
                       command=self.update_plot_visibility).pack(anchor=tk.W, pady=3)
        ttk.Checkbutton(visibility_frame, text="GT Path", variable=self.visibility_vars['gt'],
                       command=self.update_plot_visibility).pack(anchor=tk.W, pady=3)
        ttk.Checkbutton(visibility_frame, text="Basalt Path", variable=self.visibility_vars['basalt'],
                       command=self.update_plot_visibility).pack(anchor=tk.W, pady=3)
        ttk.Checkbutton(visibility_frame, text="Corrected Path", variable=self.visibility_vars['corrected'],
                       command=self.update_plot_visibility).pack(anchor=tk.W, pady=3)

        # Control buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X)

        load_btn = ttk.Button(button_frame, text="Load Data", command=self.load_data)
        load_btn.pack(fill=tk.X, pady=(0, 8), ipady=8)
        
        compute_btn = ttk.Button(button_frame, text="Compute", command=self.compute_correction)
        compute_btn.pack(fill=tk.X, pady=(0, 8), ipady=8)
        
        reset_btn = ttk.Button(button_frame, text="Reset Parameters", command=self.reset_parameters)
        reset_btn.pack(fill=tk.X, ipady=8)

        # Status label - make it more prominent
        status_frame = ttk.LabelFrame(left_panel, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(10, 0))
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="green", 
                                     font=('TkDefaultFont', 12, 'bold'))
        self.status_label.pack(fill=tk.X)

        # Plot frame
        plot_frame = ttk.Frame(right_panel)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar with zoom and pan
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        # Add mouse wheel zoom functionality (fixed direction)
        def on_scroll(event):
            if event.inaxes != self.ax:
                return
            
            # Zoom factor - scroll up zooms in, scroll down zooms out
            # event.button: 'up' = scroll up, 'down' = scroll down
            zoom_factor = 0.9 if event.button == 'up' else 1.1
            
            # Get current limits
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            
            # Get mouse position in data coordinates
            xdata = event.xdata
            ydata = event.ydata
            
            if xdata is None or ydata is None:
                return
            
            # Calculate new limits centered on mouse position
            new_width = (cur_xlim[1] - cur_xlim[0]) * zoom_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * zoom_factor
            
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            
            new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
            new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.canvas.draw()

        # Connect scroll event
        self.canvas.mpl_connect('scroll_event', on_scroll)
        
        # Add panning with middle mouse button or spacebar
        self.pan_start = None
        self.pan_axes = None
        
        def on_button_press(event):
            if event.inaxes != self.ax:
                return
            # Middle mouse button or spacebar + left click for panning
            if event.button == 2 or (event.button == 1 and event.key == ' '):
                self.pan_start = (event.xdata, event.ydata)
                self.pan_axes = self.ax
        
        def on_button_release(event):
            self.pan_start = None
            self.pan_axes = None
        
        def on_motion(event):
            if self.pan_start is None or self.pan_axes != self.ax:
                return
            if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
                return
            
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            
            self.ax.set_xlim([cur_xlim[0] - dx, cur_xlim[1] - dx])
            self.ax.set_ylim([cur_ylim[0] - dy, cur_ylim[1] - dy])
            self.canvas.draw()
            
            self.pan_start = (event.xdata, event.ydata)
        
        self.canvas.mpl_connect('button_press_event', on_button_press)
        self.canvas.mpl_connect('button_release_event', on_button_release)
        self.canvas.mpl_connect('motion_notify_event', on_motion)
        
        # Store toolbar reference for pan functionality
        self.toolbar = toolbar

        # Initial empty plot
        self.ax.set_xlabel('X [meters]', fontsize=12)
        self.ax.set_ylabel('Y [meters]', fontsize=12)
        self.ax.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
        self.ax.legend(fontsize=10)
        self.ax.grid(True)
        self.ax.axis('equal')
        self.canvas.draw()

    def update_param_value(self, param_name, param_type, value):
        """Update parameter value from slider with proper rounding."""
        config = {
            'pose_segment_size': {'step': 10, 'type': int},
            'knn_neighbors': {'step': 5, 'type': int},
            'valid_correspondence_threshold': {'step': 0.1, 'type': float},
            'icp_error_threshold': {'step': 0.5, 'type': float},
            'trimming_ratio': {'step': 0.1, 'type': float},
            'min_distance_threshold': {'step': 2.0, 'type': float},
            'max_error_consecutive': {'step': 10, 'type': int},
        }
        
        step = config[param_name]['step']
        if param_type == int:
            # Round to nearest step for integers
            rounded_value = round(float(value) / step) * step
            self.param_vars[param_name].set(int(rounded_value))
        else:
            # Round to nearest step for floats
            rounded_value = round(float(value) / step) * step
            # Round to appropriate decimal places
            decimal_places = len(str(step).split('.')[-1]) if '.' in str(step) else 0
            self.param_vars[param_name].set(round(rounded_value, decimal_places))

    def reset_parameters(self):
        """Reset all parameters to default values."""
        for param_name, default_value in self.default_args.items():
            self.param_vars[param_name].set(float(default_value))

    def on_sequence_change(self, event=None):
        """Handle sequence selection change."""
        self.current_seq = self.seq_var.get()
        self.status_label.config(text=f"Sequence changed to {self.current_seq}. Click 'Load Data' to reload.", 
                                foreground="orange")
        # Clear cached data
        self.gt_poses = None
        self.basalt_poses = None
        self.points_lane_map = None
        self.lanelet_map = None
        self.poses_corrected = None

    def load_data(self):
        """Load KITTI and basalt data for the selected sequence."""
        try:
            self.status_label.config(text="Loading data...", foreground="orange", 
                                   font=('TkDefaultFont', 12, 'bold'))
            self.show_plot_status("Loading data...", "orange")
            self.root.update()

            seq = self.seq_var.get()
            basalt_pose_file_path = os.path.join(self.basalt_pose_dir, f"{seq}.csv")
            map_path = os.path.join(self.kitti_base_dir, "map", seq, f"{seq}_map_points.npz")
            map_lanelet_path = os.path.join(self.kitti_base_dir, "map", seq, f"lanelet2_seq_{seq}.osm")

            # Check if files exist
            if not os.path.exists(basalt_pose_file_path):
                raise FileNotFoundError(f"Basalt pose file not found: {basalt_pose_file_path}")
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"Map file not found: {map_path}")
            if not os.path.exists(map_lanelet_path):
                raise FileNotFoundError(f"Lanelet map file not found: {map_lanelet_path}")

            # Check if sequence is available
            if seq not in angle_dict:
                raise ValueError(f"Sequence {seq} is not available in angle_dict")

            # Load KITTI data
            kitti_odom = pykitti.odometry(self.kitti_base_dir, seq)

            # Transformations
            T_cam0_velo = kitti_odom.calib.T_cam0_velo
            tf_yaw_to_enu = np.eye(4)
            tf_yaw_to_enu[:3, :3] = Rotation.from_euler('z', -angle_dict[seq], degrees=True).as_matrix()

            # Ground truth poses
            gt_poses = np.array(kitti_odom.poses) @ T_cam0_velo
            gt_poses = tf_yaw_to_enu @ np.linalg.inv(gt_poses[0]) @ gt_poses

            # Basalt poses
            basalt_poses = utils.read_basalt_pose(basalt_pose_file_path) @ T_cam0_velo
            basalt_poses = tf_yaw_to_enu @ np.linalg.inv(basalt_poses[0]) @ basalt_poses

            # Load map points
            new_origin_gps = [cordinta_dict[seq]['origin_lat'], cordinta_dict[seq]['origin_lon']]
            points_lane_map = utils.get_map_points(map_path, new_origin_gps)

            # Load lanelet map
            _utm_projector = lanelet2.projection.UtmProjector(
                lanelet2.io.Origin(new_origin_gps[0], new_origin_gps[1])
            )
            lanelet_map = lanelet2.io.load(map_lanelet_path, _utm_projector)

            # Store loaded data
            self.gt_poses = gt_poses
            self.basalt_poses = basalt_poses
            self.points_lane_map = points_lane_map
            self.lanelet_map = lanelet_map
            self.current_seq = seq

            self.status_label.config(text=f"Data loaded successfully for sequence {seq}", 
                                   foreground="green", font=('TkDefaultFont', 12, 'bold'))
            self.hide_plot_status()
            self.root.update()

        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            self.status_label.config(text=error_msg, foreground="red", 
                                   font=('TkDefaultFont', 12, 'bold'))
            self.show_plot_status("Error loading data!", "red")
            messagebox.showerror("Error", error_msg)

    def compute_correction(self):
        """Compute trajectory correction and update plot."""
        if self.basalt_poses is None or self.points_lane_map is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        try:
            self.status_label.config(text="Computing correction...", foreground="orange",
                                   font=('TkDefaultFont', 12, 'bold'))
            self.show_plot_status("Computing correction...", "orange")
            self.root.update()

            # Get current parameter values
            args = {}
            param_config = {
                'pose_segment_size': int,
                'knn_neighbors': int,
                'valid_correspondence_threshold': float,
                'icp_error_threshold': float,
                'trimming_ratio': float,
                'min_distance_threshold': float,
                'max_error_consecutive': int,
            }

            for param_name, param_type in param_config.items():
                value = self.param_vars[param_name].get()
                args[param_name] = param_type(value)

            # Create corrector
            trajectory_correction = OdomCorrector(self.points_lane_map, args)

            # Apply correction to all poses
            poses_corrected = []
            for i in range(len(self.basalt_poses)):
                pose_received = self.basalt_poses[i]
                pose_corrected, message = trajectory_correction.apply(pose_received)
                poses_corrected.append(pose_corrected)

            self.poses_corrected = np.array(poses_corrected)

            # Update plot
            self.update_plot()

            self.status_label.config(text="Computation completed successfully", foreground="green",
                                   font=('TkDefaultFont', 12, 'bold'))
            self.hide_plot_status()

        except Exception as e:
            error_msg = f"Error computing correction: {str(e)}"
            self.status_label.config(text=error_msg, foreground="red",
                                   font=('TkDefaultFont', 12, 'bold'))
            self.show_plot_status("Error computing correction!", "red")
            messagebox.showerror("Error", error_msg)

    def update_plot(self):
        """Update the trajectory plot."""
        if self.poses_corrected is None:
            return

        self.ax.clear()
        # Clear status text if present
        if self.status_text:
            self.status_text = None
        # Clear plot artists
        self.plot_artists = {
            'map': [],
            'gt': None,
            'basalt': None,
            'corrected': None,
        }

        # Extract XY positions from each trajectory
        def poses_to_xy(poses):
            return poses[:, 0, 3], poses[:, 1, 3]

        gt_x, gt_y = poses_to_xy(self.gt_poses)
        basalt_x, basalt_y = poses_to_xy(self.basalt_poses)
        corr_x, corr_y = poses_to_xy(self.poses_corrected)

        # Plot lanelets
        if self.lanelet_map is not None:
            for lanelet in self.lanelet_map.laneletLayer:
                xy = [[pt.x, pt.y] for pt in lanelet.centerline]
                xy = np.array(xy)
                line = self.ax.plot(xy[:, 0], xy[:, 1], color='black', alpha=0.8, linewidth=0.9)[0]
                self.plot_artists['map'].append(line)

        # Plot trajectories
        self.plot_artists['gt'] = self.ax.plot(gt_x, gt_y, 'g-', label='GT', linewidth=2)[0]
        self.plot_artists['gt'].set_visible(self.visibility_vars['gt'].get())

        self.plot_artists['basalt'] = self.ax.plot(basalt_x, basalt_y, 'r-', label='Basalt', linewidth=2)[0]
        self.plot_artists['basalt'].set_visible(self.visibility_vars['basalt'].get())

        self.plot_artists['corrected'] = self.ax.plot(corr_x, corr_y, 'b-', label='Corrected', linewidth=2)[0]
        self.plot_artists['corrected'].set_visible(self.visibility_vars['corrected'].get())

        # Set initial visibility for map
        for line in self.plot_artists['map']:
            line.set_visible(self.visibility_vars['map'].get())

        self.ax.set_xlabel('X [meters]', fontsize=12)
        self.ax.set_ylabel('Y [meters]', fontsize=12)
        self.ax.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
        self.ax.axis('equal')
        self.ax.grid(True)

        # Update legend to only show visible items
        handles = []
        labels = []
        if self.plot_artists['gt'] and self.visibility_vars['gt'].get():
            handles.append(self.plot_artists['gt'])
            labels.append('GT')
        if self.plot_artists['basalt'] and self.visibility_vars['basalt'].get():
            handles.append(self.plot_artists['basalt'])
            labels.append('Basalt')
        if self.plot_artists['corrected'] and self.visibility_vars['corrected'].get():
            handles.append(self.plot_artists['corrected'])
            labels.append('Corrected')

        if handles:
            self.ax.legend(handles, labels, fontsize=10)

        self.canvas.draw()

    def show_plot_status(self, message, color="orange"):
        """Display a prominent status message on the plot."""
        if self.status_text:
            self.status_text.remove()
        
        # Get plot center
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        
        # Create text with background box
        self.status_text = self.ax.text(x_center, y_center, message, 
                                       fontsize=24, fontweight='bold',
                                       color=color, ha='center', va='center',
                                       bbox=dict(boxstyle='round,pad=1', 
                                                facecolor='white', 
                                                edgecolor=color, 
                                                linewidth=3, alpha=0.9))
        self.canvas.draw()

    def hide_plot_status(self):
        """Remove status message from plot."""
        if self.status_text:
            self.status_text.remove()
            self.status_text = None
            self.canvas.draw()

    def update_plot_visibility(self):
        """Update visibility of plot elements based on checkbox states."""
        if self.poses_corrected is None:
            return

        # Update map visibility
        for line in self.plot_artists['map']:
            if line:
                line.set_visible(self.visibility_vars['map'].get())

        # Update trajectory visibility
        if self.plot_artists['gt']:
            self.plot_artists['gt'].set_visible(self.visibility_vars['gt'].get())
        if self.plot_artists['basalt']:
            self.plot_artists['basalt'].set_visible(self.visibility_vars['basalt'].get())
        if self.plot_artists['corrected']:
            self.plot_artists['corrected'].set_visible(self.visibility_vars['corrected'].get())

        # Update legend to only show visible items
        handles = []
        labels = []
        if self.plot_artists['gt'] and self.visibility_vars['gt'].get():
            handles.append(self.plot_artists['gt'])
            labels.append('GT')
        if self.plot_artists['basalt'] and self.visibility_vars['basalt'].get():
            handles.append(self.plot_artists['basalt'])
            labels.append('Basalt')
        if self.plot_artists['corrected'] and self.visibility_vars['corrected'].get():
            handles.append(self.plot_artists['corrected'])
            labels.append('Corrected')

        if handles:
            self.ax.legend(handles, labels, fontsize=10)
        else:
            self.ax.legend().remove()

        self.canvas.draw()


def main():
    """Main entry point for the GUI application."""
    parser = argparse.ArgumentParser(description='Trajectory Correction Parameter Tuning GUI')
    parser.add_argument('--basalt-dir', type=str, required=True,
                       help='Base directory for basalt pose files (e.g., /path/to/basalt/)')
    parser.add_argument('--kitti-dir', type=str, required=True,
                       help='Base directory for KITTI dataset (e.g., /path/to/kitti/dataset/)')

    args = parser.parse_args()

    root = tk.Tk()
    app = TrajectoryCorrectionGUI(root, args.basalt_dir, args.kitti_dir)
    root.mainloop()


if __name__ == '__main__':
    main()

