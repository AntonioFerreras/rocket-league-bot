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

GORILLA_GLUE_INDEX = 6
FLIP_RESET_INDEX = 7
SETUP_INDEX = 8

# Ground/ball height
GROUND_Z = 93.15  # Ball radius on ground

# Max angle (in degrees) between ball-to-wall direction and perpendicular to wall
MAX_WALL_APPROACH_ANGLE = 20
MAX_WALL_APPROACH_TAN = math.tan(math.radians(MAX_WALL_APPROACH_ANGLE))

def get_wall_point_first():
    """
    Pick a wall meeting point first (left or right side wall).
    Returns the wall point and which wall side (-1 for left, +1 for right).
    """
    # Choose left or right wall randomly
    wall_side = random.choice([-1, 1])
    wall_x = wall_side * (SIDE_WALL_X - WALL_MARGIN)
    
    # Y position: anywhere along the wall (with margin from back walls)
    wall_y = random.uniform(-BACK_WALL_Y + WALL_MARGIN + 500, BACK_WALL_Y - WALL_MARGIN - 500)
    
    return np.array([wall_x, wall_y, GROUND_Z]), wall_side

def get_ground_ball_spawn_for_wall(wall_point, wall_side):
    """
    Get a random ground position for ball spawn constrained by wall point.
    The angle from ball to wall must be within 20 degrees of perpendicular to the wall.
    Ball must be at least 800 units from any wall (per axis).
    """
    ball_wall_margin = 800
    wall_x = wall_point[0]
    wall_y = wall_point[1]
    
    # Ball X range: must be 800 from walls
    ball_x_min = -SIDE_WALL_X + ball_wall_margin
    ball_x_max = SIDE_WALL_X - ball_wall_margin
    
    # Pick ball_x first - it determines how far from wall we are
    # For right wall (wall_side=1), ball must be to the left (smaller x)
    # For left wall (wall_side=-1), ball must be to the right (larger x)
    if wall_side > 0:  # Right wall
        ball_x = random.uniform(ball_x_min, ball_x_max)
    else:  # Left wall
        ball_x = random.uniform(ball_x_min, ball_x_max)
    
    # Distance in X from ball to wall
    dist_x = abs(wall_x - ball_x)
    
    # Max allowed Y deviation based on 20 degree angle constraint
    # tan(angle) = |dy| / |dx|, so |dy| <= |dx| * tan(20°)
    max_dy = dist_x * MAX_WALL_APPROACH_TAN
    
    # Ball Y must be within max_dy of wall_y, AND within field bounds
    ball_y_min = max(-BACK_WALL_Y + ball_wall_margin, wall_y - max_dy)
    ball_y_max = min(BACK_WALL_Y - ball_wall_margin, wall_y + max_dy)
    
    # Ensure valid range
    if ball_y_min > ball_y_max:
        # Fallback: just use wall_y clamped to valid range
        ball_y = max(-BACK_WALL_Y + ball_wall_margin, min(BACK_WALL_Y - ball_wall_margin, wall_y))
    else:
        ball_y = random.uniform(ball_y_min, ball_y_max)
    
    return np.array([ball_x, ball_y, GROUND_Z])

def get_car_spawn_near_ball(ball_pos):
    """
    Get a random ground position for car spawn.
    Must be within 1500 unit square of ball, on ground, not within 600 of walls.
    """
    wall_margin = 600
    spawn_radius = 750  # Half of 1500 unit square
    
    while True:
        # Random offset within 1500 unit square
        offset_x = random.uniform(-spawn_radius, spawn_radius)
        offset_y = random.uniform(-spawn_radius, spawn_radius)
        
        x = ball_pos[0] + offset_x
        y = ball_pos[1] + offset_y
        
        # Clamp to field bounds with wall margin
        if (-SIDE_WALL_X + wall_margin <= x <= SIDE_WALL_X - wall_margin and
            -BACK_WALL_Y + wall_margin <= y <= BACK_WALL_Y - wall_margin):
            return np.array([x, y, 17.0])  # Car on ground height

def get_car_spawn_inline_with_wall(ball_pos, wall_point):
    """
    Get a ground position for car spawn that is in-line with ball and wall point.
    Car is placed behind the ball (opposite direction from wall).
    Must be on ground, not within 600 of walls.
    """
    wall_margin = 600
    
    # Direction from ball to wall
    direction = wall_point[:2] - ball_pos[:2]
    dist_to_wall = np.linalg.norm(direction)
    
    if dist_to_wall < 1e-4:
        # Fallback if ball is at wall point
        return get_car_spawn_near_ball(ball_pos)
    
    direction_normalized = direction / dist_to_wall
    
    # Place car behind ball (opposite direction from wall)
    # Random distance between 300-800 units behind ball
    behind_distance = random.uniform(300, 800)
    
    car_x = ball_pos[0] - direction_normalized[0] * behind_distance
    car_y = ball_pos[1] - direction_normalized[1] * behind_distance
    
    # Clamp to field bounds with wall margin
    car_x = max(-SIDE_WALL_X + wall_margin, min(SIDE_WALL_X - wall_margin, car_x))
    car_y = max(-BACK_WALL_Y + wall_margin, min(BACK_WALL_Y - wall_margin, car_y))
    
    return np.array([car_x, car_y, 17.0])  # Car on ground height

def generate_ground_setup_points(ball_spawn, wall_point, step_distance):
    """
    Generate ground setup points from ball spawn to wall point.
    Returns list of points spaced by step_distance.
    """
    direction = wall_point - ball_spawn
    total_dist = np.linalg.norm(direction)
    
    if total_dist < step_distance:
        return [wall_point.copy()]
    
    direction_normalized = direction / total_dist
    num_points = int(total_dist / step_distance)
    
    points = []
    for i in range(1, num_points + 1):
        dist = i * step_distance
        if dist >= total_dist:
            points.append(wall_point.copy())
            break
        else:
            points.append(ball_spawn + direction_normalized * dist)
    
    # Make sure we end at the wall point
    if len(points) == 0 or np.linalg.norm(points[-1] - wall_point) > 10:
        points.append(wall_point.copy())
    
    return points

def generate_aerial_path_from_point(start_point, end_point, step_distance):
    """
    Generate aerial bezier path from a start point to an end point (goal).
    Returns path_points, control_point.
    """
    goal_y = end_point[1]
    start_y = start_point[1]
    
    # Determine valid Y range for control point
    y_min = min(start_y, goal_y)
    y_max = max(start_y, goal_y)
    
    if abs(goal_y - start_y) < 1152:
        buffer = 1400
        if goal_y > start_y:
            y_min = max(SAFE_Y_RANGE[0], start_y - buffer)
        else:
            y_max = min(SAFE_Y_RANGE[1], start_y + buffer)
    
    cp_y = random.uniform(y_min, y_max)
    
    corner_margin = WALL_MARGIN * math.sqrt(2)
    corner_limit_const = SIDE_WALL_X + BACK_WALL_Y - CORNER_CATHETUS_LENGTH - corner_margin
    max_x_from_corners = corner_limit_const - abs(cp_y)
    max_x = min(SIDE_WALL_X - WALL_MARGIN, max_x_from_corners)
    
    cp_x = random.uniform(-max_x, max_x)
    
    safe_max_z = CEILING_Z + 200
    z_sample = random.uniform(FLOOR_MARGIN, safe_max_z * 1.5)
    cp_z = min(z_sample, safe_max_z)
    
    raw_control_point = np.array([cp_x, cp_y, cp_z])
    
    path_length = np.linalg.norm(end_point - start_point)
    midpoint = (start_point + end_point) / 2
    
    min_dist = 1152.0
    max_dist = 10240.0
    min_strength = 0.38
    max_strength = 1.1
    
    if path_length <= min_dist:
        curve_strength = min_strength
    elif path_length >= max_dist:
        curve_strength = max_strength
    else:
        t_len = (path_length - min_dist) / (max_dist - min_dist)
        curve_strength = min_strength + t_len * (max_strength - min_strength)
    
    control_point = midpoint + (raw_control_point - midpoint) * curve_strength
    
    path_points = sample_points_on_path(start_point, control_point, end_point, step_distance)
    
    return path_points, raw_control_point

def generate_aerial_path_from_wall(start_point, end_point, wall_side, step_distance):
    """
    Generate aerial bezier path from a wall start point to goal.
    Control point is biased toward the start (wall) and high up for strong verticality.
    Returns path_points, control_point.
    """
    goal_y = end_point[1]
    start_y = start_point[1]
    start_x = start_point[0]
    start_z = start_point[2]
    
    # Control point Y: very close to start (wall), only 5-25% of the way to goal
    t_y = random.uniform(0.05, 0.25)
    cp_y = start_y + (goal_y - start_y) * t_y
    
    # Control point X: stay very close to wall, minimal inward movement
    # wall_side is -1 for left wall, +1 for right wall
    inward_offset = random.uniform(100, 400)  # Stay close to wall
    cp_x = start_x - wall_side * inward_offset
    
    # Clamp to field bounds
    corner_margin = WALL_MARGIN * math.sqrt(2)
    corner_limit_const = SIDE_WALL_X + BACK_WALL_Y - CORNER_CATHETUS_LENGTH - corner_margin
    max_x_from_corners = corner_limit_const - abs(cp_y)
    max_x = min(SIDE_WALL_X - WALL_MARGIN, max_x_from_corners)
    cp_x = max(-max_x, min(max_x, cp_x))
    
    # Control point Z: very high for strong verticality (biased heavily toward ceiling)
    # For setup paths, we need extra height to create a proper vertical launch from wall
    # min_z = max(start_z + 1000, 1700)  # At least 1000 above start or 1700 minimum
    # max_z = CEILING_Z + 500  # Allow very close to ceiling
    cp_z = random.uniform(2300, 3000)
    
    raw_control_point = np.array([cp_x, cp_y, cp_z])
    
    # For wall paths, anchor control point very close to start
    # This creates a strong upward launch before curving to goal
    path_length = np.linalg.norm(end_point - start_point)
    
    # Blend is very low - keep control point near start
    min_dist = 2000.0
    max_dist = 8000.0
    
    if path_length <= min_dist:
        blend = 0.15  # Control point stays very close to start
    elif path_length >= max_dist:
        blend = 0.35
    else:
        t_len = (path_length - min_dist) / (max_dist - min_dist)
        blend = 0.15 + t_len * 0.2
    
    # Blend point very close to start
    blend_point = start_point + (end_point - start_point) * blend
    
    # Control point: mostly use raw_control_point position, anchored near blend_point
    control_point = blend_point.copy()
    control_point[0] = cp_x  # Use wall-biased X
    control_point[1] = blend_point[1] * 0.3 + cp_y * 0.7  # Bias Y toward wall
    control_point[2] = cp_z  # Use high Z directly
    
    path_points = sample_points_on_path(start_point, control_point, end_point, step_distance)
    
    return path_points, raw_control_point

def generate_random_path(step_distance=1000, setup_step_distance=1000, has_setup_probability=0.5, flip_reset_probability=0.7):
    """
    Generates a path from a random point to a goal.
    30% chance: Setup path (ground -> wall -> aerial)
    70% chance: Aerial-only path (existing behavior)
    
    Returns: path_points, start_point, end_point, control_point, 
             glue_conditions, flip_reset_conditions, setup_conditions, path_info
    
    path_info dict contains:
        - has_setup: bool
        - ball_spawn: np.array (for setup paths)
        - car_spawn: np.array (for setup paths)
        - num_setup_points: int
    """
    end_point = get_random_point_in_goal()
    
    has_setup = random.random() < has_setup_probability
    path_info = {
        "has_setup": has_setup,
        "ball_spawn": None,
        "car_spawn": None,
        "num_setup_points": 0
    }
    
    if has_setup:
        # Setup path: ground -> wall -> aerial
        # Pick wall point first, then constrain ball spawn by 20 degree angle rule
        wall_point, wall_side = get_wall_point_first()
        ball_spawn = get_ground_ball_spawn_for_wall(wall_point, wall_side)
        car_spawn = get_car_spawn_inline_with_wall(ball_spawn, wall_point)
        
        # Generate ground points from ball to wall (use smaller step for setup to encourage dribbling)
        ground_points = generate_ground_setup_points(ball_spawn, wall_point, setup_step_distance)
        
        # Add wall climb point (300 units up the wall)
        wall_climb_point = wall_point.copy()
        wall_climb_point[2] = 300 + GROUND_Z
        ground_points.append(wall_climb_point)
        
        num_setup_points = len(ground_points)
        
        # Aerial start point: slightly off the wall and higher
        aerial_start = wall_climb_point.copy()
        aerial_start[0] -= wall_side * 200  # Move away from wall
        aerial_start[2] += 200  # A bit higher
        
        # Generate aerial path from wall - more vertical with control point near wall
        aerial_points, raw_control_point = generate_aerial_path_from_wall(
            aerial_start, end_point, wall_side, step_distance
        )
        
        # Combine ground + aerial points
        all_points = ground_points + list(aerial_points)
        path_points = np.array(all_points)
        start_point = ball_spawn
        
        path_info["ball_spawn"] = ball_spawn
        path_info["car_spawn"] = car_spawn
        path_info["num_setup_points"] = num_setup_points
        
    else:
        # Aerial-only path (existing behavior)
        start_point = get_random_point_in_field()
        aerial_points, raw_control_point = generate_aerial_path_from_point(
            start_point, end_point, step_distance
        )
        path_points = aerial_points
        num_setup_points = 0

    # Generate conditions for path points
    num_path_points = len(path_points)
    glue_conditions = np.zeros(num_path_points, dtype=np.float32)
    flip_reset_conditions = np.zeros(num_path_points, dtype=np.float32)
    setup_conditions = np.zeros(num_path_points, dtype=np.float32)
    
    # Mark setup points
    if has_setup and num_setup_points > 0:
        setup_conditions[:num_setup_points] = 1.0
    
    # Generate Gorilla Glue conditions (Index 6) - only for non-setup points
    aerial_start_idx = num_setup_points
    num_aerial_points = num_path_points - aerial_start_idx
    
    if num_aerial_points > 0:
        if num_aerial_points <= 5:
            val = random.choice([0.0, 1.0])
            glue_conditions[aerial_start_idx:] = val
        else:
            split_point = random.randint(0, num_aerial_points)
            first_val = random.choice([0.0, 1.0])
            second_val = 1.0 - first_val
            
            glue_conditions[aerial_start_idx:aerial_start_idx + split_point] = first_val
            glue_conditions[aerial_start_idx + split_point:] = second_val

        # 50/50 chance to clear all glue conditions
        if random.choice([True, False]):
            glue_conditions[aerial_start_idx:] = 0.0

    # Generate Flip Reset conditions (Index 7) - only for non-setup, non-glue points
    if num_aerial_points > 0:
        non_glue_indices = (glue_conditions == 0.0) & (setup_conditions == 0.0)
        if np.any(non_glue_indices):
            if random.random() < flip_reset_probability:
                flip_reset_conditions[non_glue_indices] = 1.0
                # First aerial point can't have flip reset
                if aerial_start_idx < num_path_points:
                    flip_reset_conditions[aerial_start_idx] = 0.0

    return path_points, start_point, end_point, raw_control_point, glue_conditions, flip_reset_conditions, setup_conditions, path_info


