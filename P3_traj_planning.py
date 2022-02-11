import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch # Switch occurs at t_final - t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        # Hint: Both self.traj_controller and self.pose_controller have compute_control() functions. 
        #       When should each be called? Make use of self.t_before_switch and 
        #       self.traj_controller.traj_times.
        ########## Code starts here ##########
        if t < (self.traj_controller.traj_times[-1] - self.t_before_switch):
            return self.traj_controller.compute_control(x,y,th,t)
        else:
            return self.pose_controller.compute_control(x,y,th,t)
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.
    dist_between_points = [np.linalg.norm(np.array(path[i+1])-np.array(path[i])) for i in range(len(path)-1)]
    
    nominal_time = [int(np.ceil(dist/V_des)) for dist in dist_between_points]
    
    total_time = np.sum(nominal_time)
    t_smoothed = np.linspace(0, total_time, int(total_time/dt))
    
    time_pts = np.array([np.sum(nominal_time[:i]) for i in range(len(nominal_time)+1)])
    
    x_pts = np.array([loc[0] for loc in path])
    y_pts = np.array([loc[1] for loc in path])
    
    spl_x = scipy.interpolate.splrep(time_pts, x_pts, s=alpha)
    spl_y = scipy.interpolate.splrep(time_pts, y_pts, s=alpha)
    
    x_d = scipy.interpolate.splev(t_smoothed, spl_x)
    y_d = scipy.interpolate.splev(t_smoothed, spl_y)
    
    xd_d = scipy.interpolate.splev(t_smoothed, spl_x, der=1)
    yd_d = scipy.interpolate.splev(t_smoothed, spl_y, der=1)

    xdd_d = scipy.interpolate.splev(t_smoothed, spl_x, der=2)
    ydd_d = scipy.interpolate.splev(t_smoothed, spl_y, der=2)
    
    theta_d = np.arctan2(yd_d,xd_d)    
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    Hint: Take a close look at the code within compute_traj_with_limits() and interpolate_traj() 
          from P1_differential_flatness.py
    """
    ########## Code starts here ##########
    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V,t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    s_f = State(x=traj[-1,0], y = traj[-1,1], V= V_tilde[-1], th=traj[-1,2])
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)    
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
