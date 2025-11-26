import tkinter as tk
import numpy as np
import random
import math

# Field constants from rlgym/rocket_league/common_values.py
SIDE_WALL_X = 4096
BACK_WALL_Y = 5120
BACK_NET_Y = 6000  # Added for goal visualization
CEILING_Z = 2044
GOAL_HEIGHT = 642.775
GOAL_CENTER_TO_POST = 892.755
CORNER_CATHETUS_LENGTH = 1152

# Margins
FLOOR_MARGIN = 500
CEILING_MARGIN = 150
WALL_MARGIN = 100

# Derived constants
FIELD_X_RANGE = (-SIDE_WALL_X, SIDE_WALL_X)
FIELD_Y_RANGE = (-BACK_WALL_Y, BACK_WALL_Y)
FIELD_Z_RANGE = (0, CEILING_Z)

# Safe ranges
SAFE_X_RANGE = (-SIDE_WALL_X + WALL_MARGIN, SIDE_WALL_X - WALL_MARGIN)
SAFE_Y_RANGE = (-BACK_WALL_Y + WALL_MARGIN, BACK_WALL_Y - WALL_MARGIN)
SAFE_Z_RANGE = (FLOOR_MARGIN, CEILING_Z - CEILING_MARGIN)

# Goal locations (Y planes)
GOAL_Y_BLUE = -BACK_WALL_Y
GOAL_Y_ORANGE = BACK_WALL_Y

def is_point_in_safe_field(p):
    x, y, z = p
    # Check bounding box
    if not (SAFE_X_RANGE[0] <= x <= SAFE_X_RANGE[1]): return False
    if not (SAFE_Y_RANGE[0] <= y <= SAFE_Y_RANGE[1]): return False
    if not (SAFE_Z_RANGE[0] <= z <= SAFE_Z_RANGE[1]): return False
    
    # Check corners (Rocket League field has 45-degree corners)
    # With wall margin, the corner wall moves inward by margin * sqrt(2)
    corner_margin = WALL_MARGIN * math.sqrt(2)
    limit = SIDE_WALL_X + BACK_WALL_Y - CORNER_CATHETUS_LENGTH - corner_margin
    
    if abs(x) + abs(y) > limit:
        return False
        
    return True

def get_random_point_in_field():
    # Critical points: Center, Left Post, Right Post for both goals
    goal_points = [
        np.array([0, -BACK_WALL_Y]),
        np.array([-GOAL_CENTER_TO_POST, -BACK_WALL_Y]),
        np.array([GOAL_CENTER_TO_POST, -BACK_WALL_Y]),
        np.array([0, BACK_WALL_Y]),
        np.array([-GOAL_CENTER_TO_POST, BACK_WALL_Y]),
        np.array([GOAL_CENTER_TO_POST, BACK_WALL_Y])
    ]

    while True:
        x = random.uniform(*SAFE_X_RANGE)
        y = random.uniform(*SAFE_Y_RANGE)
        z = random.uniform(*SAFE_Z_RANGE)
        p = np.array([x, y, z])
        
        # Check proximity to goal points in XY
        p_xy = np.array([x, y])
        
        # Find closest critical point
        dists = [np.linalg.norm(p_xy - gp) for gp in goal_points]
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        min_dist_threshold = 2500
        
        if min_dist < min_dist_threshold:
            closest_pt = goal_points[min_dist_idx]
            # Push point away to exactly 1000 distance in same direction
            vec = p_xy - closest_pt
            norm = np.linalg.norm(vec)
            
            if norm < 1e-4:
                continue # Too close to determine direction, retry
                
            new_vec = (vec / norm) * min_dist_threshold
            new_xy = closest_pt + new_vec
            
            p[0] = new_xy[0]
            p[1] = new_xy[1]
            
        if is_point_in_safe_field(p):
            return p

def get_random_point_in_goal():
    # Pick a side randomly
    is_orange = random.choice([True, False])
    y = GOAL_Y_ORANGE if is_orange else GOAL_Y_BLUE
    
    # Random X within goal width (with margin)
    x_limit = GOAL_CENTER_TO_POST - WALL_MARGIN
    x = random.uniform(-x_limit, x_limit)
    
    # Random Z within goal height (with margin)
    # Z must be > FLOOR_MARGIN and < GOAL_HEIGHT - WALL_MARGIN (top post)
    z_min = FLOOR_MARGIN
    z_max = GOAL_HEIGHT - WALL_MARGIN
    z = random.uniform(z_min, z_max)
    
    return np.array([x, y, z])

def quadratic_bezier(p0, p1, p2, t):
    """
    Calculate point on quadratic bezier curve for time t in [0,1].
    B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    """
    return (1-t)**2 * p0 + 2 * (1-t) * t * p1 + t**2 * p2

def sample_points_on_path(start_point, control_point, end_point, step_distance):
    """
    Sample roughly evenly spaced points along the quadratic bezier curve.
    Does not include the start point.
    N is computed based on path length and step_distance.
    """
    # To get evenly spaced points, we need to estimate the arc length.
    # Oversample factor - use fixed high count since we don't know N yet
    oversample = 1000
    t_vals = np.linspace(0, 1, oversample)
    points = np.array([quadratic_bezier(start_point, control_point, end_point, t) for t in t_vals])
    
    # Calculate cumulative distance
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_dists = np.concatenate(([0], np.cumsum(dists)))
    total_length = cumulative_dists[-1]
    
    # Determine Number of Points (N)
    if step_distance <= 10: step_distance = 10 # Safety
    
    # "N points computed from that scale based on path distance"
    num_samples = int(total_length / step_distance)
    if num_samples < 1: num_samples = 1
    
    # Target distances
    target_dists = np.linspace(total_length / num_samples, total_length, num_samples)
    
    final_points = []
    for d in target_dists:
        # Find the index where cumulative distance >= d
        idx = np.searchsorted(cumulative_dists, d)
        if idx == 0:
            final_points.append(points[0]) 
        elif idx >= len(points):
            final_points.append(points[-1])
        else:
            # Interpolate between idx-1 and idx
            d0 = cumulative_dists[idx-1]
            d1 = cumulative_dists[idx]
            if d1 == d0:
                 final_points.append(points[idx])
            else:
                ratio = (d - d0) / (d1 - d0)
                p = points[idx-1] + (points[idx] - points[idx-1]) * ratio
                final_points.append(p)
            
    return np.array(final_points)

def generate_random_path(step_distance=1000):
    """
    Generates a smooth random path (quadratic bezier) from a random point
    in the field to a random point in a goal.
    Uses direct sampling for the control point to satisfy constraints without loops.
    Returns sampled points along the path (excluding start).
    """
    start_point = get_random_point_in_field()
    # This function randomly selects between Blue and Orange goals
    end_point = get_random_point_in_goal()
    
    # Determine goal direction (Y-axis)
    goal_y = end_point[1]
    start_y = start_point[1]
    
    # Determine valid Y range for control point
    # Base range: strictly monotonic approach (between start and goal)
    y_min = min(start_y, goal_y)
    y_max = max(start_y, goal_y)
    
    # Exception: If start is close to target backboard, allow moving away slightly
    # "start point is within 1152 Y of the backboard for that goal"
    if abs(goal_y - start_y) < 1152:
        buffer = 1400
        if goal_y > start_y: # Targeting Orange (Positive Y)
            # Can go back (lower Y)
            # Clamp to safe range
            y_min = max(SAFE_Y_RANGE[0], start_y - buffer)
        else: # Targeting Blue (Negative Y)
            # Can go back (higher Y)
            y_max = min(SAFE_Y_RANGE[1], start_y + buffer)
    
    # Sample Y directly from valid range
    cp_y = random.uniform(y_min, y_max)
    
    # Determine valid X range for control point (accounting for truncated corners)
    # Corner constraint with margin
    corner_margin = WALL_MARGIN * math.sqrt(2)
    corner_limit_const = SIDE_WALL_X + BACK_WALL_Y - CORNER_CATHETUS_LENGTH - corner_margin
    
    max_x_from_corners = corner_limit_const - abs(cp_y)
    
    # Also bounded by side walls (safe range)
    max_x = min(SIDE_WALL_X - WALL_MARGIN, max_x_from_corners)
    
    # Sample X directly
    cp_x = random.uniform(-max_x, max_x)
    
    # Sample Z directly
    # Allow curve to go higher (1.5x more likely to reach near ceiling)
    # We expand the range we pick from, then clamp it
    # Base safe max Z
    safe_max_z = CEILING_Z - CEILING_MARGIN
    z_sample = random.uniform(FLOOR_MARGIN, safe_max_z * 1.5)
    cp_z = min(z_sample, safe_max_z) 
    
    # Create the random control point
    raw_control_point = np.array([cp_x, cp_y, cp_z])
    
    # "Make the amount of curve scaled by path length."
    # "Very short paths within the backboard distance are curve a quarter as much"
    # "Very long full court long paths curve 1x much"
    
    # Calculate straight-line distance
    path_length = np.linalg.norm(end_point - start_point)
    midpoint = (start_point + end_point) / 2
    
    # Define reference lengths
    # "within backboard distance" ~ 1152 units (Short)
    # "full court" ~ Length of field = 10240 units (Long)
    
    min_dist = 1152.0
    max_dist = 10240.0
    
    # Strength factors
    min_strength = 0.25 # Quarter as much
    max_strength = 1.0  # 1x much
    
    # Linear interpolation of strength based on length
    if path_length <= min_dist:
        curve_strength = min_strength
    elif path_length >= max_dist:
        curve_strength = max_strength
    else:
        # Lerp
        t_len = (path_length - min_dist) / (max_dist - min_dist)
        curve_strength = min_strength + t_len * (max_strength - min_strength)

    control_point = midpoint + (raw_control_point - midpoint) * curve_strength
    
    # Sample roughly evenly spaced points
    path_points = sample_points_on_path(start_point, control_point, end_point, step_distance)
    print(path_points)
    return path_points, start_point, end_point, raw_control_point

class PathVisualizer:
    def __init__(self, blocking=True):
        self.root = tk.Tk()
        self.root.title("Rocket League Path Generator (Tkinter)")
        
        # Window setup
        self.width = 800
        self.height = 900
        self.root.geometry(f"{self.width}x{self.height}")
        
        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height - 50, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Control Frame for Button and Input
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, pady=10)

        # Button
        self.btn = tk.Button(control_frame, text="New Path", command=self.update_path, font=("Arial", 12))
        self.btn.pack(side=tk.LEFT, padx=10)
        
        # Step Distance Input
        tk.Label(control_frame, text="Step Dist:").pack(side=tk.LEFT, padx=5)
        self.step_dist_var = tk.StringVar(value="1000")
        self.entry_step_dist = tk.Entry(control_frame, textvariable=self.step_dist_var, width=5)
        self.entry_step_dist.pack(side=tk.LEFT)
        
        if blocking:
            self.update_path()
            self.root.mainloop()

    def process_events(self):
        self.root.update()
        self.root.update_idletasks()

    def show_path(self, path, start, end, control):
        self.canvas.delete("all")
        
        self.draw_field_top_down()
        self.draw_field_side_view()
        
        # Draw Path Top-Down
        td_points = [self.transform_top_down(p[0], p[1]) for p in path]
        # Flatten list of tuples for create_line
        td_coords = [coord for point in td_points for coord in point]
        if len(td_coords) >= 4:
            self.canvas.create_line(td_coords, fill="red", width=2, smooth=True)

        # Draw points in yellow
        for px, py in td_points:
            self.canvas.create_oval(px-2, py-2, px+2, py+2, fill="yellow", outline="yellow")
        
        # Start/End/Control Top-Down
        sx, sy = self.transform_top_down(start[0], start[1])
        self.canvas.create_oval(sx-4, sy-4, sx+4, sy+4, fill="green")
        
        ex, ey = self.transform_top_down(end[0], end[1])
        self.canvas.create_line(ex-5, ey-5, ex+5, ey+5, fill="red", width=2)
        self.canvas.create_line(ex-5, ey+5, ex+5, ey-5, fill="red", width=2)
        
        cx, cy = self.transform_top_down(control[0], control[1])
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill="gray")
        
        # Draw Path Side View
        sv_points = [self.transform_side_view(p[1], p[2]) for p in path]
        sv_coords = [coord for point in sv_points for coord in point]
        if len(sv_coords) >= 4:
            self.canvas.create_line(sv_coords, fill="red", width=2, smooth=True)

        # Draw points in yellow (Side View)
        for px, py in sv_points:
            self.canvas.create_oval(px-2, py-2, px+2, py+2, fill="yellow", outline="yellow")
        
        # Start/End/Control Side View
        sx, sy = self.transform_side_view(start[1], start[2])
        self.canvas.create_oval(sx-4, sy-4, sx+4, sy+4, fill="green")
        
        ex, ey = self.transform_side_view(end[1], end[2])
        self.canvas.create_line(ex-5, ey-5, ex+5, ey+5, fill="red", width=2)
        self.canvas.create_line(ex-5, ey+5, ex+5, ey-5, fill="red", width=2)
        
        cx, cy = self.transform_side_view(control[1], control[2])
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill="gray")

    def transform_top_down(self, x, y):
        # Map Field X/Y to Screen X/Y
        # Field X: -4096 to 4096 -> Screen X
        # Field Y: -6000 to 6000 -> Screen Y (Included nets)
        
        scale = 0.05  # Reduced Scale factor to fit everything
        cx, cy = 400, 300  # Center of Top-Down View
        
        screen_x = cx + x * scale
        screen_y = cy + y * scale  # +y is down in screen, +y is orange in game.
        return screen_x, screen_y

    def transform_side_view(self, y, z):
        # Map Field Y/Z to Screen X/Y
        # Field Y: -6000 to 6000 -> Screen X
        # Field Z: 0 to 2044 -> Screen Y
        
        scale = 0.05  # Reduced Scale factor to fit everything
        cx, cy = 400, 750 # Center of Side View (Y axis)
        
        screen_x = cx + y * scale
        screen_y = cy - z * scale # Invert Z because screen Y is down
        return screen_x, screen_y

    def draw_field_top_down(self):
        # Draw Title
        self.canvas.create_text(150, 20, text="Top-Down View (X vs Y)", font=("Arial", 14, "bold"))
        
        # Draw Field Boundary
        # Outline coordinates (CCW from Top Right)
        # Corners are cut: (SideX, BackY-Corner) -> (SideX-Corner, BackY)
        points = [
            (SIDE_WALL_X, BACK_WALL_Y - CORNER_CATHETUS_LENGTH), # Orange Right Corner Start
            (SIDE_WALL_X - CORNER_CATHETUS_LENGTH, BACK_WALL_Y), # Orange Right Corner End
            (-SIDE_WALL_X + CORNER_CATHETUS_LENGTH, BACK_WALL_Y), # Orange Left Corner Start
            (-SIDE_WALL_X, BACK_WALL_Y - CORNER_CATHETUS_LENGTH), # Orange Left Corner End
            (-SIDE_WALL_X, -BACK_WALL_Y + CORNER_CATHETUS_LENGTH), # Blue Left Corner Start
            (-SIDE_WALL_X + CORNER_CATHETUS_LENGTH, -BACK_WALL_Y), # Blue Left Corner End
            (SIDE_WALL_X - CORNER_CATHETUS_LENGTH, -BACK_WALL_Y), # Blue Right Corner Start
            (SIDE_WALL_X, -BACK_WALL_Y + CORNER_CATHETUS_LENGTH), # Blue Right Corner End
        ]
        
        screen_points = [self.transform_top_down(p[0], p[1]) for p in points]
        self.canvas.create_polygon(screen_points, outline="black", fill="#e0e0e0", width=2)
        
        # Draw Goals (Top Down)
        # Orange Goal (Top)
        og_x1, og_y1 = self.transform_top_down(-GOAL_CENTER_TO_POST, BACK_WALL_Y)
        og_x2, og_y2 = self.transform_top_down(GOAL_CENTER_TO_POST, BACK_NET_Y)
        self.canvas.create_rectangle(og_x1, og_y1, og_x2, og_y2, outline="orange", width=2)

        # Blue Goal (Bottom)
        bg_x1, bg_y1 = self.transform_top_down(-GOAL_CENTER_TO_POST, -BACK_WALL_Y)
        bg_x2, bg_y2 = self.transform_top_down(GOAL_CENTER_TO_POST, -BACK_NET_Y)
        self.canvas.create_rectangle(bg_x1, bg_y1, bg_x2, bg_y2, outline="blue", width=2)

    def draw_field_side_view(self):
        # Draw Title
        self.canvas.create_text(100, 580, text="Side View (Y vs Z)", font=("Arial", 14, "bold"))
        
        # Draw Floor
        f_start = self.transform_side_view(-BACK_WALL_Y, 0)
        f_end = self.transform_side_view(BACK_WALL_Y, 0)
        self.canvas.create_line(f_start[0], f_start[1], f_end[0], f_end[1], fill="black", width=2)
        
        # Draw Ceiling
        c_start = self.transform_side_view(-BACK_WALL_Y, CEILING_Z)
        c_end = self.transform_side_view(BACK_WALL_Y, CEILING_Z)
        self.canvas.create_line(c_start[0], c_start[1], c_end[0], c_end[1], fill="black", width=2)
        
        # Draw End Walls
        w_blue_start = self.transform_side_view(-BACK_WALL_Y, 0)
        w_blue_end = self.transform_side_view(-BACK_WALL_Y, CEILING_Z)
        self.canvas.create_line(w_blue_start[0], w_blue_start[1], w_blue_end[0], w_blue_end[1], fill="black", width=2)
        
        w_orange_start = self.transform_side_view(BACK_WALL_Y, 0)
        w_orange_end = self.transform_side_view(BACK_WALL_Y, CEILING_Z)
        self.canvas.create_line(w_orange_start[0], w_orange_start[1], w_orange_end[0], w_orange_end[1], fill="black", width=2)

        # Draw Goals (Side View)
        # Orange Goal
        og_y_start = BACK_WALL_Y
        og_y_end = BACK_NET_Y
        og_z_top = GOAL_HEIGHT
        
        p1 = self.transform_side_view(og_y_start, 0)
        p2 = self.transform_side_view(og_y_end, 0)
        p3 = self.transform_side_view(og_y_end, og_z_top)
        p4 = self.transform_side_view(og_y_start, og_z_top)
        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="orange", width=2)
        self.canvas.create_line(p2[0], p2[1], p3[0], p3[1], fill="orange", width=2)
        self.canvas.create_line(p3[0], p3[1], p4[0], p4[1], fill="orange", width=2)
        
        # Blue Goal
        bg_y_start = -BACK_WALL_Y
        bg_y_end = -BACK_NET_Y
        bg_z_top = GOAL_HEIGHT
        
        p1 = self.transform_side_view(bg_y_start, 0)
        p2 = self.transform_side_view(bg_y_end, 0)
        p3 = self.transform_side_view(bg_y_end, bg_z_top)
        p4 = self.transform_side_view(bg_y_start, bg_z_top)
        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="blue", width=2)
        self.canvas.create_line(p2[0], p2[1], p3[0], p3[1], fill="blue", width=2)
        self.canvas.create_line(p3[0], p3[1], p4[0], p4[1], fill="blue", width=2)


    def update_path(self):
        self.canvas.delete("all")
        
        self.draw_field_top_down()
        self.draw_field_side_view()
        
        try:
            step_dist = float(self.step_dist_var.get())
        except ValueError:
            step_dist = 1000
            
        path, start, end, control = generate_random_path(step_distance=step_dist)
        
        # Draw Path Top-Down
        td_points = [self.transform_top_down(p[0], p[1]) for p in path]
        # Flatten list of tuples for create_line
        td_coords = [coord for point in td_points for coord in point]
        if len(td_coords) >= 4:
            self.canvas.create_line(td_coords, fill="red", width=2, smooth=True)

        # Draw points in yellow
        for px, py in td_points:
            self.canvas.create_oval(px-2, py-2, px+2, py+2, fill="yellow", outline="yellow")
        
        # Start/End/Control Top-Down
        sx, sy = self.transform_top_down(start[0], start[1])
        self.canvas.create_oval(sx-4, sy-4, sx+4, sy+4, fill="green")
        
        ex, ey = self.transform_top_down(end[0], end[1])
        self.canvas.create_line(ex-5, ey-5, ex+5, ey+5, fill="red", width=2)
        self.canvas.create_line(ex-5, ey+5, ex+5, ey-5, fill="red", width=2)
        
        cx, cy = self.transform_top_down(control[0], control[1])
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill="gray")
        
        # Draw Path Side View
        sv_points = [self.transform_side_view(p[1], p[2]) for p in path]
        sv_coords = [coord for point in sv_points for coord in point]
        self.canvas.create_line(sv_coords, fill="red", width=2, smooth=True)

        # Draw points in yellow (Side View)
        for px, py in sv_points:
            self.canvas.create_oval(px-2, py-2, px+2, py+2, fill="yellow", outline="yellow")
        
        # Start/End/Control Side View
        sx, sy = self.transform_side_view(start[1], start[2])
        self.canvas.create_oval(sx-4, sy-4, sx+4, sy+4, fill="green")
        
        ex, ey = self.transform_side_view(end[1], end[2])
        self.canvas.create_line(ex-5, ey-5, ex+5, ey+5, fill="red", width=2)
        self.canvas.create_line(ex-5, ey+5, ex+5, ey-5, fill="red", width=2)
        
        cx, cy = self.transform_side_view(control[1], control[2])
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill="gray")

    def update_ball(self, position):
        x, y, z = position
        
        # Top Down
        screen_x, screen_y = self.transform_top_down(x, y)
        r = 6 # Radius
        if self.canvas.find_withtag("ball_top_down"):
            self.canvas.coords("ball_top_down", screen_x-r, screen_y-r, screen_x+r, screen_y+r)
            self.canvas.tag_raise("ball_top_down")
        else:
            self.canvas.create_oval(screen_x-r, screen_y-r, screen_x+r, screen_y+r, fill="purple", outline="black", tags="ball_top_down")

        # Side View
        screen_x_side, screen_y_side = self.transform_side_view(y, z)
        if self.canvas.find_withtag("ball_side_view"):
            self.canvas.coords("ball_side_view", screen_x_side-r, screen_y_side-r, screen_x_side+r, screen_y_side+r)
            self.canvas.tag_raise("ball_side_view")
        else:
             self.canvas.create_oval(screen_x_side-r, screen_y_side-r, screen_x_side+r, screen_y_side+r, fill="purple", outline="black", tags="ball_side_view")

if __name__ == "__main__":
    viz = PathVisualizer()
