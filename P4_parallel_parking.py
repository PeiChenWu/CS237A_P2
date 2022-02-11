import numpy as np
import matplotlib.pyplot as plt
import reeds_shepp
from utils import plot_line_segments, line_line_intersection

from P2_rrt import RRT


def reeds_shepp_path_sample_with_cusps(q0, q1, turning_radius, step_size):
    """Generates points along a Reeds-Shepp path sampled (i) at a regular interval and (ii) including cusp points.

    NOTE: You may find this function useful for most accurate collision checking (otherwise the cusps between forward/
    backward maneuvers may be ignored -- this is often where the most critical tolerances are for parallel parking!).

    Args:
        q0: Array-like of 3 numbers `(x0, y0, th0)`; the initial configuration.
        q1: Array-like of 3 numbers `(x1, y1, th1)`; the final configuration.
        turning_radius: Scalar; the turning radius of the vehicle.
        step_size: Scalar, the arclength interval along the path at which to sample.

    Returns:
        A list of 3-tuples `(x, y, th)` sampled along the Reeds-Shepp path at interval `step_size` that is sure to also
        include `q0`, `q1`, and any transition points (possibly cusps) between Reeds-Shepp path segments.
    """
    cusp_distances = np.cumsum(
        [abs(length) for _, length in reeds_shepp.path_type(q0, q1, turning_radius) if abs(length) > 1e-2])[:-1]
    return [
        q for d, q in sorted(
            [(step_size * i, q[:3]) for i, q in enumerate(reeds_shepp.path_sample(q0, q1, turning_radius, step_size))] +
            [(d, reeds_shepp.path_sample(q0, q1, turning_radius, d)[1][:3]) for d in cusp_distances if d > 1e-2])
    ] + [q1]


class ParkingRRT(RRT):
    """Represents a planning problem for the Reeds-Shepp car with rectangular geometry.

    See `self.body_corners` and `self.body_corners_at_config` to understand what workspace footprint a given
    configuration corresponds to.

    The Reeds-Shepp car moves at a constant speed forward or backward and has a limited turning radius. We will use the
    `reeds_shepp` package at https://github.com/liespace/pyReedsShepp/ to compute steering distances and steering
    trajectories. See http://planning.cs.uiuc.edu/node822.html for more details on deriving these steering trajectories.
    """

    def __init__(self,
                 statespace_lo,
                 statespace_hi,
                 x_init,
                 x_goal,
                 obstacles,
                 turning_radius=10.0,
                 half_length=2.0,
                 half_width=0.9,
                 rear_axle_to_center_distance=1.0):
        self.turning_radius = turning_radius
        self.half_length = half_length
        self.half_width = half_width
        self.rear_axle_to_center_distance = rear_axle_to_center_distance
        self.body_corners = np.array([[half_length + rear_axle_to_center_distance, -half_width],
                                      [half_length + rear_axle_to_center_distance, half_width],
                                      [-half_length + rear_axle_to_center_distance, half_width],
                                      [-half_length + rear_axle_to_center_distance, -half_width],
                                      [half_length + rear_axle_to_center_distance, -half_width]])
        super().__init__(statespace_lo, statespace_hi, x_init, x_goal, obstacles)

    def body_corners_at_config(self, q):
        """Returns an array containing the corners of the car body (in the workspace) at configuration `q`.

        Args:
            q: Array-like of 3 numbers `(x0, y0, th0)`; a configuration of the Reeds-Shepp car.

        Returns:
            An array of shape (5, 2) containing the x/y-coordinates of the corners of the car's body, specifically:
                [
                    [front right x, front right y],
                    [ front left x,  front left y],
                    [  back left x,   back left y],
                    [ back right x,  back right y],
                    [front right x, front right y]
                ]
            (NOTE: this array contains the front right corner repeated at the start and the end so that extracting
            edges, e.g., in `ParkingRRT.plot_problem` below, is simpler -- to be clear, the body geometry is still a
            rectangle with four corners).
        """
        return np.array(q[:2]) + self.body_corners @ np.array([[np.cos(q[2]), np.sin(q[2])],
                                                               [-np.sin(q[2]), np.cos(q[2])]])

    def plot_tree(self, V, P, resolution=np.pi / 60, **kwargs):
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                qs = reeds_shepp_path_sample_with_cusps(V[P[i], :], V[i, :], self.turning_radius,
                                                        self.turning_radius * resolution)
                for j in range(len(qs) - 1):
                    line_segments.append((qs[j], qs[j + 1]))
        kwargs["alpha"] = 0.2
        plot_line_segments(line_segments, **kwargs)

    def plot_path(self, resolution=np.pi / 60, **kwargs):
        qs = []
        path = np.array(self.path)
        for i in range(path.shape[0] - 1):
            new_qs = reeds_shepp_path_sample_with_cusps(path[i], path[i + 1], self.turning_radius,
                                                        self.turning_radius * resolution)
            qs.extend(new_qs)
        plt.plot([x for x, y, th in qs], [y for x, y, th in qs], **kwargs)
        if kwargs.get("color") == "green":
            line_segments = []
            for q in qs:
                body_corners = self.body_corners_at_config(q)
                line_segments += [body_corners[i:i + 2] for i in range(4)]
            plot_line_segments(line_segments, color="green", linewidth=1, alpha=0.5)

    def plot_problem(self):
        super().plot_problem()
        plot_line_segments([self.body_corners_at_config(self.x_init)[i:i + 2] for i in range(4)])
        plot_line_segments([self.body_corners_at_config(self.x_goal)[i:i + 2] for i in range(4)])
        plt.axis("equal")

    def find_nearest(self, V, x):
        ########## Code starts here ##########
        p_len_lst = []
        for state in np.array(V):
            p_len_lst.append(reeds_shepp.path_length(np.array(state),np.array(x), self.turning_radius))
        return V[np.argmin(p_len_lst)]
        ########## Code ends here ##########

    def steer_towards(self, x1, x2, eps):
        """
        A subtle issue: if you use d_path.sample_many to return the point
        at distance eps along the path from x to y, use a turning radius
        slightly larger than self.turning_radius
        (i.e., 1.001*self.turning_radius). Without this hack,
        d_path.sample_many might return a point that can't quite get to in
        distance eps (using self.turning_radius) due to numerical precision
        issues.
        """
        ########## Code starts here ##########
        rs_path = reeds_shepp_path_sample_with_cusps(np.array(x1), np.array(x2), 1.001*self.turning_radius, eps)
        return rs_path[1]
        ########## Code ends here ##########

    def is_free_motion(self, obstacles, x1, x2, resolution = np.pi/12):
        rs_path = reeds_shepp_path_sample_with_cusps(x1,x2, self.turning_radius, self.turning_radius * resolution)
        for i in range(len(rs_path) - 1):
            for line in obstacles:
                corners = self.body_corners_at_config(rs_path[i])
                for j in range(len(corners)-1):
                    if line_line_intersection([corners[j][:2], corners[j+1][:2]], line):
                        return False
        return True

    ### Override base `RRT` methods here as necessary, i.e., look at the methods in the base `RRT` class in P2_rrt.py
    # and see which ones need to be changed/reimplemented for `ParkingRRT` (HINT: possibly see `P2_rrt.GeometricRRT`
    # and `P2_rrt.DubinsRRT` for inspiration). HINT: How will you validate collision-free motions in this case? You may
    # find `line_line_intersection` from utils.py useful as well as `reeds_shepp_path_sample_with_cusps` above.
