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
CEILING_MARGIN = 350
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
    # Place it deep in the goal (halfway between goal line and back net)
    goal_depth = BACK_NET_Y - BACK_WALL_Y
    y_offset = goal_depth / 2
    y = (GOAL_Y_ORANGE + y_offset) if is_orange else (GOAL_Y_BLUE - y_offset)
    
    # Random X within goal width (with margin)
    x_limit = GOAL_CENTER_TO_POST - WALL_MARGIN
    x = random.uniform(-x_limit, x_limit)
    
    # Random Z within goal height (with margin)
    # Z must be > 200 and < GOAL_HEIGHT - WALL_MARGIN (top post)
    z_min = 200
    z_max = GOAL_HEIGHT + 200 # - WALL_MARGIN
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
    safe_max_z = CEILING_Z + 200
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
    min_strength = 0.38
    max_strength = 1.1
    
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

    # Generate Gorilla Glue conditions (Index 6)
    num_path_points = len(path_points)
    glue_conditions = np.zeros(num_path_points, dtype=np.float32)
    
    if num_path_points > 0:
        if num_path_points <= 5:
            val = random.choice([0.0, 1.0])
            glue_conditions[:] = val
        else:
            split_point = random.randint(0, num_path_points)
            first_val = random.choice([0.0, 1.0])
            second_val = 1.0 - first_val
            
            glue_conditions[:split_point] = first_val
            glue_conditions[split_point:] = second_val
    
    return path_points, start_point, end_point, raw_control_point, glue_conditions


