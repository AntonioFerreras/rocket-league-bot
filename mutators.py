import random
import numpy as np
from typing import Dict, Any
from rlgym.rocket_league.api import GameState
from rlgym.api import StateMutator
from rlgym.rocket_league import common_values
from math_utils import dir_to_euler_yzx, normalize

class AirDribbleMutator(StateMutator[GameState]):
    """
    A StateMutator that sets up the game state for a kickoff.
    """


    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        state.config.boost_consumption = 0.001

        # spawn_min_x = -3200
        # spawn_max_x = 3200

        # spawn_min_y = -2000
        # spawn_max_y = 3600
        
        # spawn_min_z = 700
        # spawn_max_z = 1800

        # car_min_height_under_ball = 100 # 500
        # car_max_height_under_ball = 800 # 1000
        # car_x_radius = 50 # 800
        # car_y_min = -50 # -800
        # car_y_max = 50 # -800
        # car_dir_noise_radius = 0*np.pi/16

        # ball_vel_xy_noise_radius = 400
        # ball_vel_z_min = 50
        # ball_vel_z_max = 650

        # car_vel_noise_radius = 0
        # car_speed_min = 300
        # car_speed_max = 400
        spawn_min_x = -3200
        spawn_max_x = 3200

        spawn_min_y = -2000
        spawn_max_y = 3600
        
        spawn_min_z = 1000
        spawn_max_z = 1500

        car_min_height_under_ball = 100 # 500
        car_max_height_under_ball = 350 # 1000
        car_x_radius = 20 # 800
        car_y_min = 20 # -800
        car_y_max = 20 # -800
        car_dir_noise_radius = 0*np.pi/16

        ball_vel_xy_noise_radius = 100
        ball_vel_z_min = 150
        ball_vel_z_max = 550

        car_vel_noise_radius = 0
        car_speed_min = 300
        car_speed_max = 400


        ball_x = random.uniform(spawn_min_x, spawn_max_x)
        ball_y = random.uniform(spawn_min_y, spawn_max_y)
        ball_z = random.uniform(spawn_min_z, spawn_max_z)

        ball_vel = np.array([
            random.uniform(-ball_vel_xy_noise_radius, ball_vel_xy_noise_radius), 
            random.uniform(-ball_vel_xy_noise_radius, ball_vel_xy_noise_radius), 
            random.uniform(ball_vel_z_min, ball_vel_z_max)
        ], dtype=np.float32)

        state.ball.position = np.array([ball_x, ball_y, ball_z], dtype=np.float32)

        # add a component of ball velocity in direction of the goal
        objective = np.array(common_values.ORANGE_GOAL_BACK) - state.ball.position
        objective = objective / np.linalg.norm(objective)
        objective[2] = 0.0
        ball_vel = ball_vel + objective * random.uniform(100, 800)*0

        state.ball.linear_velocity = ball_vel
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        

        for car in state.cars.values():
            pos_x = ball_x + random.uniform(-car_x_radius, car_x_radius)
            pos_y = ball_y + random.uniform(car_y_min, car_y_max) # make car behind ball most of the time
            pos_z = ball_z - random.uniform(car_min_height_under_ball, car_max_height_under_ball)

            # clamp car pos to be within spawn limits
            pos_x = max(spawn_min_x, min(pos_x, spawn_max_x))
            pos_y = max(spawn_min_y, min(pos_y, spawn_max_y))
            pos_z = max(400, min(pos_z, spawn_max_z))


            car.physics.position = np.array([pos_x, pos_y, pos_z], dtype=np.float32)

            to_ball = state.ball.position - car.physics.position + random.uniform(-car_vel_noise_radius, car_vel_noise_radius)
            to_ball = to_ball / np.linalg.norm(to_ball)
            car_vel = to_ball * random.uniform(car_speed_min, car_speed_max)

            car.physics.linear_velocity = car_vel
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            # Aim car toward the ball
            to_ball = state.ball.position - car.physics.position 
            car.physics.euler_angles = dir_to_euler_yzx(to_ball) + random.uniform(-car_dir_noise_radius, car_dir_noise_radius)
            car.boost_amount = 100.0
            car.air_time_since_jump = 2.0 # start with no flip
            car.has_jumped = True

        
        shared_info["target_x"] = min(max(state.ball.position[0] + random.uniform(-200, 200), -common_values.SIDE_WALL_X + 100), common_values.SIDE_WALL_X - 100)
        shared_info["target_y"] = min(max(state.ball.position[1] + random.uniform(-200, 200), -common_values.BACK_WALL_Y + 100), common_values.BACK_WALL_Y - 100)
        shared_info["target_z"] = min(max(state.ball.position[2] + random.uniform(-200, 200), 700), 1600)


class AirDribbleDirectedMutator(AirDribbleMutator):
    """
    An AirDribbleMutator that generates a directed path for the ball.
    """

    def __init__(self, num_conditions: int):
        super().__init__()
        self.num_conditions = num_conditions

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        from path_generator import generate_random_path
        
        # Call super to set up basic car/ball state, but we will override position/velocity
        super().apply(state, shared_info)
        
        # Generate a path
        path_points, start_point, end_point, control_point, glue_conditions = generate_random_path(step_distance=1000)

        num_path_points = len(path_points)
        condition_data = np.zeros((num_path_points, self.num_conditions), dtype=np.float32)

        # Set 7th flag (index 6) based on generated glue conditions
        if num_path_points > 0:
            condition_data[:, 6] = glue_conditions
        
        
        # Override Ball Position to be the Start Point
        state.ball.position = start_point.astype(np.float32)
        
        if len(path_points) > 0:
            first_target = path_points[0]
        else:
            first_target = end_point
            
        objective_direction = normalize(first_target - start_point)
        objective_direction[2] = 0.0
        
        ball_vel_xy_noise_radius = 50
        ball_vel_z_min = 150
        ball_vel_z_max = 550


        ball_vel = np.array([
            random.uniform(-ball_vel_xy_noise_radius, ball_vel_xy_noise_radius), 
            random.uniform(-ball_vel_xy_noise_radius, ball_vel_xy_noise_radius), 
            random.uniform(ball_vel_z_min, ball_vel_z_max)
        ], dtype=np.float32)
        state.ball.linear_velocity = ball_vel + objective_direction * random.uniform(150, 300)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        
        # Store path points in shared_info
        shared_info["path_points"] = path_points.astype(np.float32)
        shared_info["path_start"] = start_point.astype(np.float32)
        shared_info["path_end"] = end_point.astype(np.float32)
        shared_info["path_control"] = control_point.astype(np.float32)

        shared_info["air_roll_rate"] = 0.0
        shared_info["air_roll_action"] = 0
        shared_info["condition_data"] = condition_data
        
        # Re-position cars based on the new ball position (Start Point)
        # This reuses the logic from AirDribbleMutator but applies it to the new ball location
        # We can largely copy the car positioning logic or call a helper if we refactored
        # For now, let's just re-run the car positioning part locally to ensure it's relative to the NEW ball pos
        
        FLOOR_MARGIN = 400
        CEILING_MARGIN = 450
        WALL_MARGIN = 100


        # Use safe ranges from path_generator
        spawn_min_x = -common_values.SIDE_WALL_X + WALL_MARGIN
        spawn_max_x = common_values.SIDE_WALL_X - WALL_MARGIN
        spawn_min_y = -common_values.BACK_WALL_Y + WALL_MARGIN
        spawn_max_y = common_values.BACK_WALL_Y - WALL_MARGIN
        spawn_min_z = FLOOR_MARGIN
        spawn_max_z = common_values.CEILING_Z - CEILING_MARGIN
        
        car_min_height_under_ball = 100 
        car_max_height_under_ball = 400
        car_x_radius = 20
        car_y_min = 20
        car_y_max = 20
        car_vel_noise_radius = 0
        car_speed_min = 300
        car_speed_max = 400
        car_dir_noise_radius = 0
        
        ball_x, ball_y, ball_z = state.ball.position

        for car in state.cars.values():
            pos_x = ball_x + random.uniform(-car_x_radius, car_x_radius)
            pos_y = ball_y + random.uniform(car_y_min, car_y_max) # make car behind ball most of the time
            pos_z = ball_z - random.uniform(car_min_height_under_ball, car_max_height_under_ball)

            # clamp car pos to be within spawn limits
            pos_x = max(spawn_min_x, min(pos_x, spawn_max_x))
            pos_y = max(spawn_min_y, min(pos_y, spawn_max_y))
            pos_z = max(spawn_min_z, min(pos_z, spawn_max_z))

            car.physics.position = np.array([pos_x, pos_y, pos_z], dtype=np.float32)

            to_ball = state.ball.position - car.physics.position + random.uniform(-car_vel_noise_radius, car_vel_noise_radius)
            to_ball = to_ball / np.linalg.norm(to_ball)
            ball_horizontal_vel = state.ball.linear_velocity.copy()
            ball_horizontal_vel[2] = 0.0
            car_vel = ball_horizontal_vel * random.uniform(car_speed_min, car_speed_max) + ball_horizontal_vel * random.uniform(0.4, 0.9)
            car_vel = to_ball * random.uniform(car_speed_min, car_speed_max)

            car.physics.linear_velocity = car_vel
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            # Aim car toward the ball
            to_ball = state.ball.position - car.physics.position 
            car.physics.euler_angles = dir_to_euler_yzx(to_ball) + random.uniform(-car_dir_noise_radius, car_dir_noise_radius)
            car.boost_amount = 100.0
            car.air_time_since_jump = 2.0 # start with no flip
            car.has_jumped = True

        # Update target info for observation (if needed, though we have path_points now)
        shared_info["current_target_index"] = 0

