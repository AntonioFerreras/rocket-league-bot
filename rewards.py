import numpy as np
from typing import List, Dict, Any, Callable
import math
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_RADIUS, CAR_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BACK_NET_Y, SIDE_WALL_X, CEILING_Z, CAR_MAX_ANG_VEL, TICKS_PER_SECOND

from math_utils import normalize, cosine_similarity

from path_generator import GORILLA_GLUE_INDEX, FLIP_RESET_INDEX, SETUP_INDEX

def height_sigmoid(height: float) -> float:
    return 0.5 * np.tanh((height - 900) / 250) + 0.5

class GoalReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward of 1 if the agent's team scored a goal, -1 if the opposing team scored a goal,
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> float:
        # No goal reward during setup
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        if condition_data[SETUP_INDEX] > 0.5:
            return 0.0
        
        dist_path_end_to_ball = np.linalg.norm(state.ball.position - shared_info["path_points"][-1])
        dist_car_to_ball = np.linalg.norm(state.cars[agent].physics.position - state.ball.position)
        if state.cars[agent].physics.position[2] < 60.0 or dist_path_end_to_ball > 2300.0 or dist_car_to_ball > 550.0:
            return 0.0
        return state.goal_scored
            


class BoostChangeReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, gain_weight: float = 0.0, lose_weight=1.0,
                 activation_fn: Callable[[float], float] = lambda x: math.sqrt(0.01 * x)):
        """
        Reward function that rewards agents for increasing their boost and penalizes them for decreasing it.

        :param gain_weight: Weight to apply to the reward when the agent gains boost
        :param lose_weight: Weight to apply to the reward when the agent loses boost
        :param activation_fn: Activation function to apply to the boost value before calculating the reward. Default is
                              the square root function so that increasing boost is more important when boost is low.
        """
        self.gain_weight = gain_weight
        self.lose_weight = lose_weight
        self.activation_fn = activation_fn

        self.prev_values = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_values = {
            agent: self.activation_fn(initial_state.cars[agent].boost_amount)
            for agent in agents
        }

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            current_value = self.activation_fn(state.cars[agent].boost_amount)
            delta = current_value - self.prev_values[agent]
            if delta > 0:
                rewards[agent] = delta * self.gain_weight
            elif delta < 0:
                rewards[agent] = delta * self.lose_weight
            else:
                rewards[agent] = 0
            self.prev_values[agent] = current_value

        return rewards

class DistancePlayerToBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> float:
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]

        in_setup = condition_data[SETUP_INDEX] > 0.5

        if not in_setup and (state.cars[agent].physics.position[2] < 100.0 or state.ball.position[2] < 150.0):
            return 0.0

        # Compensate for inside of ball being unreachable (keep max reward at 1)

        if condition_data[GORILLA_GLUE_INDEX] < 0.5:
            return 0.0

        dist = np.linalg.norm(state.cars[agent].physics.position - (state.ball.position)) - BALL_RADIUS
        dist_reward = 0.2*np.exp(-100.0 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

        # height_reward = height_sigmoid(state.cars[agent].physics.position[2])
        return dist_reward #* height_reward



class VelocityPlayerToBallReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward for velocity of car towards ball.
    Can use trajectory comparison or dot quotient.
    No reward when car is on the ground.
    """
    def __init__(self, include_negative_values: bool = True, use_trajectory_comparison: bool = True,
                 use_dot_quotient: bool = False):
        self.include_negative_values = include_negative_values
        self.use_trajectory_comparison = use_trajectory_comparison
        self.use_dot_quotient = use_dot_quotient

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]):
        ball = state.ball
        car = state.cars[agent].physics

        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]

        in_setup = condition_data[SETUP_INDEX] > 0.5

        if not in_setup and (car.position[2] < 100.0 or ball.position[2] < 150.0):
            return 0.0

        if self.use_trajectory_comparison:
            curr_dist, min_dist, t = trajectory_comparison(car.position, car.linear_velocity,
                                                           ball.position, ball.linear_velocity)
            vel = (curr_dist - min_dist) / t if t != 0 else 0
            norm_vel = vel / (CAR_MAX_SPEED + BALL_MAX_SPEED)
            if abs(norm_vel) > 1:  # In case of floating point errors with small t
                norm_vel = np.sign(norm_vel)
        elif self.use_dot_quotient:
            car_to_ball = ball.position - car.position
            car_to_ball = car_to_ball / np.linalg.norm(car_to_ball)

            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d which becomes v . d / |d|^2
            # Max value should be max_speed / ball_radius = 2300 / 92.75 = 24.8
            vd = np.dot(car_to_ball, car.linear_velocity)
            dd = np.dot(car_to_ball, ball.linear_velocity)
            inv_time = vd / dd if dd != 0 else 0
            norm_vel = inv_time / (CAR_MAX_SPEED / BALL_RADIUS)
        else:
            car_to_ball = ball.position - car.position
            car_to_ball = car_to_ball / np.linalg.norm(car_to_ball)

            vel = np.dot(car_to_ball, car.linear_velocity)
            norm_vel = vel / CAR_MAX_SPEED
        if self.include_negative_values:
            return norm_vel
        vel_reward = max(0, norm_vel)
        #height_reward = height_sigmoid(state.cars[agent].physics.position[2])
        return vel_reward #* height_reward

def trajectory_comparison(pos1, vel1, pos2, vel2, check_bounds=True):
    """
    Calculate the closest point between two trajectories, defined as the lines:
      pos1 + t * vel1
      pos2 + t * vel2
    """
    # First, find max time based on field bounds
    if check_bounds:
        max_time = np.inf
        for pos, vel in (pos1, vel1), (pos2, vel2):
            bounds = np.array([[-SIDE_WALL_X, -BACK_WALL_Y, 0],
                               [SIDE_WALL_X, BACK_WALL_Y, CEILING_Z]])
            times = (bounds - pos) / (vel + (vel == 0))
            times = times[times > 0]
            t = np.min(times)
            max_time = min(max_time, t)

    # The distance between the two rays is `||pos1 + t * vel1 - pos2 - t * vel2||`
    # This is equivalent to `||(pos1 - pos2) + t * (vel1 - vel2)||`
    pos_diff = pos1 - pos2
    vel_diff = vel1 - vel2

    # The minimum distance is achieved when the derivative of the distance is 0.
    # E.g. `d/dt * sqrt((p_x+t*v_x)^2+(p_y+t*v_y)^2+(p_z+t*v_z)^2)=0`
    # This is equivalent to
    #    `d/dt * (p_x+t*v_x)^2+(p_y+t*v_y)^2+(p_z+t*v_z)^2=0`
    # => `2*(p_x+t*v_x)*v_x+2*(p_y+t*v_y)*v_y+2*(p_z+t*v_z)*v_z=0`
    # => `p_x*v_x+p_y*v_y+p_z*v_z+t*(v_x^2+v_y^2+v_z^2)=0`
    # => `t=-(p_x*v_x+p_y*v_y+p_z*v_z)/(v_x^2+v_y^2+v_z^2)`
    denom = np.dot(vel_diff, vel_diff)
    if denom == 0:
        t = 0
    else:
        t = -np.dot(pos_diff, vel_diff) / denom

    if t > max_time:
        t = max_time

    # The minimum distance is then the distance at this time.
    curr_dist = np.linalg.norm(pos_diff)
    min_dist = np.linalg.norm(pos_diff + t * vel_diff)

    return curr_dist, min_dist, t

class BallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward for the ball being close to the goal.
    Also a reward for travelling towards the goal.
    """
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        player = state.cars[agent]
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        # Compensate for moving objective to back of net
        dist = np.linalg.norm(state.ball.position - objective) - (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
        dist_reward = np.exp(-0.8 * dist / BALL_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

        vel_normalized = state.ball.linear_velocity / BALL_MAX_SPEED
        goal_dir = normalize(objective - state.ball.position)
        vel_reward = max(0, np.dot(vel_normalized, goal_dir))

        return dist_reward + vel_reward

class BallToTargetReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward when the ball comes within a certain distance of the target.
    Also a reward for travelling towards the target.
    Once within the distance, increment the current target index.
    """
    def __init__(self, target_distance=600, print_hits=False):
        super().__init__()
        self.target_distance = target_distance
        self.print_hits = print_hits

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        shared_info["current_target_index"] = 0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> float:
        current_idx = shared_info["current_target_index"]
        objective = shared_info["path_points"][current_idx]
        condition_data = shared_info["condition_data"][current_idx]
        dir_target = normalize(objective - state.ball.position)
        dot_product = np.dot(dir_target, state.ball.linear_velocity/BALL_MAX_SPEED)
        dist = np.linalg.norm(state.ball.position - objective)
        car_ball_dist = np.linalg.norm(state.cars[agent].physics.position - state.ball.position)
        close_enough_to_ball = 1.0 if car_ball_dist < 500.0 else 0.0

        in_setup = condition_data[SETUP_INDEX] > 0.5
        
        # During setup, use smaller target distance (must be more precise)
        effective_target_distance = 450.0 if in_setup else self.target_distance

        if dist < effective_target_distance:
            
            # Track setup targets hit
            if in_setup:
                shared_info["num_setup_targets_hit"] = shared_info.get("num_setup_targets_hit", 0) + 1
                
                # Check if this is the LAST setup point (next point is aerial)
                next_idx = min(current_idx + 1, len(shared_info["path_points"]) - 1)
                next_condition = shared_info["condition_data"][next_idx]
                is_last_setup_point = next_condition[SETUP_INDEX] < 0.5

                if self.print_hits:
                    print(f"Hit target {current_idx} | close enough to ball: {close_enough_to_ball} | is last setup point: {is_last_setup_point}")
                
                if is_last_setup_point:
                    # Big bonus for completing setup - encourages the bot to actually hit the wall
                    hit_target_reward = 25.0
                else:
                    hit_target_reward = 1.0
            elif shared_info["condition_data"][0][SETUP_INDEX] > 0.5:
                shared_info["num_air_targets_hit_after_setup"] = shared_info.get("num_air_targets_hit_after_setup", 0) + 1
                hit_target_reward = 1.0
            else:
                hit_target_reward = 1.0
            shared_info["current_target_index"] = min(current_idx + 1, len(shared_info["path_points"]) - 1)
        else:
            # if self.print_hits:
            #     print(f"Not hit target {shared_info['current_target_index']}")
            hit_target_reward = 0.0
        
        # During setup, reduce the ball velocity component to discourage hitting the ball hard
        # Instead, SetupDribbleReward handles the continuous dribbling incentive
        velocity_multiplier = 1.0 if not in_setup else 0.5
        reward = (hit_target_reward + dot_product * 5.0 * velocity_multiplier) * close_enough_to_ball
        
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        if condition_data[GORILLA_GLUE_INDEX] < 0.5 and condition_data[FLIP_RESET_INDEX] < 0.5 and condition_data[SETUP_INDEX] < 0.5:
            roll_rate = AirRollReward.get_air_roll_rate(state, agent)
            reward *= abs(roll_rate)
        if condition_data[FLIP_RESET_INDEX] > 0.5:
            reward *= 0.35
        return reward


class ForwardBiasReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward for moving in the direction the car is facing.
    Only active during setup phase.
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(
        self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]
    ) -> float:
        # Only active during setup
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        if condition_data[SETUP_INDEX] < 0.5:
            return 0.0
        
        return state.cars[agent].physics.forward.dot(normalize(state.cars[agent].physics.linear_velocity))

class ZoneReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a punishment when agent is close to a wall, ceiling, or ground.
    Gives reward when in a good height range.
    To prevent the agent from driving on walls to avoid low height punishment.
    Disabled during setup phase and for a grace period after setup ends.
    """
    def __init__(self, grace_period_seconds: float = 2.5):
        self.grace_period_seconds = grace_period_seconds
        self.setup_ended_tick = None
        self.was_in_setup = False

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.setup_ended_tick = None
        # Check if path starts with setup
        if len(shared_info.get("condition_data", [])) > 0:
            self.was_in_setup = shared_info["condition_data"][0][SETUP_INDEX] > 0.5
        else:
            self.was_in_setup = False

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(
        self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]
    ) -> float:
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        in_setup = condition_data[SETUP_INDEX] > 0.5
        
        # Disabled during setup
        if in_setup:
            self.was_in_setup = True
            return 0.0
        
        # Track when setup ended and apply grace period
        if self.was_in_setup and self.setup_ended_tick is None:
            self.setup_ended_tick = state.tick_count
        
        if self.setup_ended_tick is not None:
            time_since_setup = (state.tick_count - self.setup_ended_tick) / TICKS_PER_SECOND
            if time_since_setup < self.grace_period_seconds:
                return 0.0  # Grace period - no punishment yet
        
        thresh = 200.0
        height_reward_weight = 0.0
        close_to_wall = np.abs(state.cars[agent].physics.position[0]) > SIDE_WALL_X - thresh
        close_to_wall = close_to_wall or np.abs(state.cars[agent].physics.position[1]) > BACK_WALL_Y - thresh
        close_to_wall = close_to_wall or state.cars[agent].physics.position[2] > CEILING_Z - thresh
        close_to_wall = close_to_wall or state.cars[agent].physics.position[2] < thresh
        if close_to_wall:
            return -2.5
        height = state.cars[agent].physics.position[2]
        height_reward = height_sigmoid(height)
        return height_reward*height_reward_weight

class BallZoneReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a punishment when ball is close to a wall, ceiling, or ground.
    Gives reward when in a good height range.
    To prevent the ball from driving on walls to avoid low height punishment.
    Disabled during setup phase and for a grace period after setup ends.
    """
    def __init__(self, grace_period_seconds: float = 2.5):
        self.grace_period_seconds = grace_period_seconds
        self.setup_ended_tick = None
        self.was_in_setup = False

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.setup_ended_tick = None
        # Check if path starts with setup
        if len(shared_info.get("condition_data", [])) > 0:
            self.was_in_setup = shared_info["condition_data"][0][SETUP_INDEX] > 0.5
        else:
            self.was_in_setup = False

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(
        self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]
    ) -> float:
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        in_setup = condition_data[SETUP_INDEX] > 0.5
        
        # Disabled during setup
        if in_setup:
            self.was_in_setup = True
            return 0.0
        
        # Track when setup ended and apply grace period
        if self.was_in_setup and self.setup_ended_tick is None:
            self.setup_ended_tick = state.tick_count
        
        if self.setup_ended_tick is not None:
            time_since_setup = (state.tick_count - self.setup_ended_tick) / TICKS_PER_SECOND
            if time_since_setup < self.grace_period_seconds:
                return 0.0  # Grace period - no punishment yet
        
        thresh_ceiling = 110.0
        thresh_floor = 150.0
        thresh_wall = 150.0
        height_reward_weight = 0.0
        close_to_wall = np.abs(state.ball.position[0]) > SIDE_WALL_X - thresh_wall
        close_to_wall = close_to_wall or np.abs(state.ball.position[1]) > BACK_WALL_Y - thresh_wall
        close_to_wall = close_to_wall or state.ball.position[2] > CEILING_Z - thresh_ceiling
        close_to_wall = close_to_wall or state.ball.position[2] < thresh_floor
        if close_to_wall:
            return -1.5
        height = state.ball.position[2]
        height_reward = height_sigmoid(height)
        return height_reward*height_reward_weight

class TouchReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward when agent touches ball.
    The more beneath the ball it hits, the higher the reward.
    Also gives an optional reward for accelerating the ball upward.
    """

    def __init__(self, acceleration_reward: float = 1.0):
        self.acceleration_reward = acceleration_reward
        self.prev_ball = None
        self.last_target_hit_tick = None
        self.last_touch_tick = None
        self.prev_car_vel = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ball = initial_state.ball
        self.prev_car_vel = None
        self.last_target_hit_tick = initial_state.tick_count
        self.current_target_index = 0
        self.last_touch_tick = None


    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        if self.current_target_index != shared_info["current_target_index"]:
            self.last_target_hit_tick = state.tick_count
            self.current_target_index = shared_info["current_target_index"]
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> float:

        if self.prev_car_vel is None:
            self.prev_car_vel = state.cars[agent].physics.linear_velocity

        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        
        # Disabled during setup
        if condition_data[SETUP_INDEX] > 0.5:
            self.prev_ball = state.ball
            self.prev_car_vel = state.cars[agent].physics.linear_velocity
            return 0.0
        
        hit_ball = 1. if state.cars[agent].ball_touches > 0 else 0.

        if hit_ball > 0:
            self.last_touch_tick = state.tick_count
        touch_reward_cooldown_active = False
        if self.last_touch_tick is not None:
            touch_reward_cooldown_active = (state.tick_count - self.last_touch_tick) / TICKS_PER_SECOND < 0.8

        if state.cars[agent].physics.position[2] < 100.0 or state.ball.position[2] < 150.0:
            return 0.0

        if (state.tick_count - self.last_target_hit_tick) / TICKS_PER_SECOND > 2:
            return 0.0

        to_ball = state.ball.position - state.cars[agent].physics.position
        to_ball = to_ball / np.linalg.norm(to_ball)
        vertical = to_ball[2]
        vertical = max(0.0, min(vertical, 0.7071))/0.7071 

        # measure how much upward velocity direction it gave the ball
        acceleration = (state.ball.linear_velocity - self.prev_ball.linear_velocity) / BALL_MAX_SPEED
        accel_dir_z = max(0.0, acceleration[2]) * 30.0
        # if hit_ball > 0:
        #     print(f"accel_dir_z: {accel_dir_z}")
        #     print(f"accel_dir_z^2: {accel_dir_z**2}")

        # if condition_data[FLIP_RESET_INDEX] > 0.5:
        #     return 0.0

        if condition_data[GORILLA_GLUE_INDEX] > 0.5:
            reward = state.cars[agent].ball_touches * vertical*.2
        elif condition_data[FLIP_RESET_INDEX] < 0.5:
            
            reward = 0.0 # accel_dir_z*accel_dir_z * hit_ball * 0.25
            # roll_rate = AirRollReward.get_air_roll_rate(state, agent)
            # reward *= abs(roll_rate)
            # dont apply touch punishment if on last path point
            if shared_info["current_target_index"] == len(shared_info["path_points"]) - 1:
                hit_ball = 0.0
            reward -= hit_ball*state.cars[agent].ball_touches*2.0
            # reward += 10.0*(self.prev_car_vel[2] - self.prev_ball.linear_velocity[2]) / BALL_MAX_SPEED
            # if state.cars[agent].ball_touches > 0:
            #     print(f"reward: {reward}, vel_diff: {10.0*(self.prev_car_vel[2] - self.prev_ball.linear_velocity[2]) / BALL_MAX_SPEED}")
            # if touch_reward_cooldown_active:
            #     reward *= 0.2
            shared_info["hit_accel_dir_z"] = accel_dir_z
            shared_info["num_ball_touches"] = state.cars[agent].ball_touches
            
            
        else:
            reward = -hit_ball*state.cars[agent].ball_touches*2.0
            
        
        self.prev_ball = state.ball
        self.prev_car_vel = state.cars[agent].physics.linear_velocity
        
        return reward

class AirRollReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward for using air roll.
    """
    def __init__(self):
        self.last_target_hit_tick = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.current_target_index = 0
        self.last_target_hit_tick = initial_state.tick_count

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        if self.current_target_index != shared_info["current_target_index"]:
            self.last_target_hit_tick = state.tick_count
            self.current_target_index = shared_info["current_target_index"]
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    @staticmethod
    def get_air_roll_rate(state: GameState, agent: AgentID) -> float:
        R = state.cars[agent].physics.rotation_mtx
        omega_world = state.cars[agent].physics.angular_velocity
        omega_body = omega_world @ R
        roll_rate = omega_body[0] / CAR_MAX_ANG_VEL
        return roll_rate

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> float:
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]

        # Disabled during setup
        if condition_data[SETUP_INDEX] > 0.5:
            return 0.0

        if (state.tick_count - self.last_target_hit_tick) / TICKS_PER_SECOND > 2:
            return 0.0
        roll_rate = self.get_air_roll_rate(state, agent)
        # yaw_rate = abs(omega_body[1]) / 9.11
        shared_info["air_roll_rate"] = abs(roll_rate)
        air_roll_action = -shared_info["air_roll_action"]

        if condition_data[GORILLA_GLUE_INDEX] > 0.5:
            return -1.0 if air_roll_action != 0 else 0.0
        if condition_data[FLIP_RESET_INDEX] > 0.5:
            return 0.0

        if roll_rate > 0.0: roll_rate = 1.0
        elif roll_rate < 0.0: roll_rate = -1.0
        else: roll_rate = 0.0

        dist_to_ball = np.linalg.norm(state.cars[agent].physics.position - state.ball.position)
        if dist_to_ball > 250.0:
            return 0.0
        else:
            return roll_rate * air_roll_action

        return roll_rate * air_roll_action


        # if roll_rate * air_roll_action > 0:
        #     rol_is_aligned = 1.0
        # else:

        #     rol_is_aligned = 0.0
        # print(f"roll_rate: {roll_rate:.4f}, air_roll_action: {air_roll_action}")
        # return rol_is_aligned

class FlipResetReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, debug: bool = False, obtain_flip_weight: float = 6.0, hit_ball_weight: float = 0.0, down_facing_ball_weight: float = 0.8, target_hit_timeout: float = 1.5):
        self.debug = debug
        self.obtain_flip_weight = obtain_flip_weight
        self.hit_ball_weight = hit_ball_weight
        self.down_facing_ball_weight = down_facing_ball_weight
        self.target_hit_timeout = target_hit_timeout
        self.last_target_hit_tick = None

        self.prev_state = None
        self.has_reset = None
        self.has_flipped = None
        self.down_facing_juice = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state
        self.has_reset = set()
        self.has_flipped = set()
        self.down_facing_juice = self.obtain_flip_weight*0.5
        self.last_target_hit_tick = initial_state.tick_count
        self.current_target_index = 0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        if self.current_target_index != shared_info["current_target_index"]:
            self.last_target_hit_tick = state.tick_count
            self.current_target_index = shared_info["current_target_index"]
        rewards = {k: 0.0 for k in agents}
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]

        # Disabled during setup
        if condition_data[SETUP_INDEX] > 0.5:
            self.prev_state = state
            return rewards

        if condition_data[FLIP_RESET_INDEX] < 0.5:
            return rewards

        if (state.tick_count - self.last_target_hit_tick) / TICKS_PER_SECOND > self.target_hit_timeout:
            return rewards
        

        

        for agent in agents:
            car = state.cars[agent]
            down = -car.physics.up
            car_ball = state.ball.position - car.physics.position
            cossim_down_ball = max(0, cosine_similarity(down, car_ball))
            distance_to_goal = np.linalg.norm(state.cars[agent].physics.position - shared_info["path_points"][-1])
            distance_scale = min(1.0 + 1.5*max(0.0, distance_to_goal - 2300.0)/2300.0, 4.0)
            pointing_in_motion_direction = np.dot(normalize(state.cars[agent].physics.forward[:2]), normalize(state.cars[agent].physics.linear_velocity[:2])) > 0.766044443
            if car.ball_touches > 0 and car.has_flip and not self.prev_state.cars[agent].has_flip:
                if cossim_down_ball > 0.5 ** 0.5 - 0.2:  # 45 degrees
                    self.has_reset.add(agent)
                    rewards[agent] = 0.0
                    
                    shared_info["num_flip_resets"] += 1
                    shared_info["reset_distance_to_goal"] = distance_to_goal

                    
                    multi_scale = 2.0 if shared_info["num_flip_resets"] > 1 else 1.0
                    first_reset_ball_vel_scale = 0.5 + (min(200, max(-600, self.prev_state.ball.linear_velocity[2])) + 600) * (1.0 / 800)
                    pointing_in_motion_direction_scale = (0.5 if (not pointing_in_motion_direction and shared_info["num_flip_resets"] == 1) else 1.0)
                    if shared_info["num_flip_resets"] > 0:
                        first_reset_ball_vel_scale = 1.0
                    if state.cars[agent].physics.position[2] > 80.0:
                        rewards[agent] += self.obtain_flip_weight*distance_scale*multi_scale*first_reset_ball_vel_scale#*pointing_in_motion_direction_scale

                    if self.debug:
                        print(f"Flip reset {shared_info["num_flip_resets"]} distance to goal {shared_info["reset_distance_to_goal"]}")
                        print(f"Ball z velocity: {self.prev_state.ball.linear_velocity[2]}")
            elif car.on_ground:
                self.has_reset.discard(agent)
                self.has_flipped.discard(agent)
            elif car.is_flipping and agent in self.has_reset:
                self.has_reset.remove(agent)
                self.has_flipped.add(agent)
                self.down_facing_juice = self.obtain_flip_weight*0.5
            if car.ball_touches > 0 and agent in self.has_flipped:
                self.has_flipped.remove(agent)
                rewards[agent] = self.hit_ball_weight
                # if self.debug:
                #     print(f"Hit ball {shared_info["num_flip_resets"]}")
            
            # dist to ball decreasing while down is facing ball
            prev_dist_to_ball = np.linalg.norm(self.prev_state.cars[agent].physics.position - self.prev_state.ball.position) - BALL_RADIUS
            curr_dist_to_ball = np.linalg.norm(car.physics.position - state.ball.position) - BALL_RADIUS
            dist_decreasing = prev_dist_to_ball > curr_dist_to_ball
            close_to_ground = state.cars[agent].physics.position[2] < 80.0 or state.ball.position[2] < 150.0
            if close_to_ground:
                self.down_facing_juice = 0.0
            decrease_rate = (prev_dist_to_ball - curr_dist_to_ball)

            # if prev_dist_to_ball > curr_dist_to_ball:
            if curr_dist_to_ball < 100 and self.down_facing_juice > 0.0 and decrease_rate > 7.0:
                
                close_reward = self.down_facing_ball_weight * (cossim_down_ball*2.0 - 1.0) # * (0.5 if (not pointing_in_motion_direction and shared_info["num_flip_resets"] == 0) else 1.0)
                if decrease_rate > 30.0 and close_reward > 0.0:
                    close_reward *= 2.0
                    
                self.down_facing_juice -= close_reward
                
                rewards[agent] += close_reward*distance_scale
        self.prev_state = state
        return rewards


class SetupJumpFlipPunishment(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that punishes jumping and flipping during setup phase.
    """
    def __init__(self, jump_punishment: float = -1.0, flip_punishment: float = -2.0):
        self.jump_punishment = jump_punishment
        self.flip_punishment = flip_punishment
        self.prev_has_jumped = {}
        self.prev_is_flipping = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_has_jumped = {agent: initial_state.cars[agent].has_jumped for agent in agents}
        self.prev_is_flipping = {agent: initial_state.cars[agent].is_flipping for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> float:
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        
        # Only active during setup
        if condition_data[SETUP_INDEX] < 0.5:
            self.prev_has_jumped[agent] = state.cars[agent].has_jumped
            self.prev_is_flipping[agent] = state.cars[agent].is_flipping
            return 0.0
        
        reward = 0.0
        car = state.cars[agent]
        
        # Punish for starting a jump (transition from not jumped to jumped)
        if car.has_jumped and not self.prev_has_jumped.get(agent, False):
            reward += self.jump_punishment
        
        # Punish for starting a flip (transition from not flipping to flipping)
        if car.is_flipping and not self.prev_is_flipping.get(agent, False):
            reward += self.flip_punishment
        
        self.prev_has_jumped[agent] = car.has_jumped
        self.prev_is_flipping[agent] = car.is_flipping
        
        return reward


class SetupDribbleReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that rewards keeping the ball close while moving toward the target.
    Only active during setup phase. Encourages actual dribbling instead of hitting the ball ahead.
    Tapers off as ball approaches the wall to encourage committing to the final hit.
    """
    def __init__(self, close_distance: float = 250.0, max_ball_speed_diff: float = 400.0):
        """
        :param close_distance: Distance threshold to consider ball "on" the car for dribbling
        :param max_ball_speed_diff: Max speed difference between ball and car to be considered dribbling
        """
        self.close_distance = close_distance
        self.max_ball_speed_diff = max_ball_speed_diff

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> float:
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        
        # Only active during setup
        if condition_data[SETUP_INDEX] < 0.5:
            return 0.0
        
        car = state.cars[agent]
        ball = state.ball
        
        # Distance from car to ball
        car_to_ball = ball.position - car.physics.position
        dist_to_ball = np.linalg.norm(car_to_ball)
        
        # Reward for ball being close (exponential falloff)
        # Max reward when dist=0, falls off as distance increases
        close_reward = np.exp(-dist_to_ball / self.close_distance)
        
        # Get target direction
        current_idx = shared_info["current_target_index"]
        objective = shared_info["path_points"][current_idx]
        to_target = normalize(objective - car.physics.position)
        
        # Reward for car velocity toward target
        car_vel_toward_target = np.dot(car.physics.linear_velocity, to_target)
        car_speed = np.linalg.norm(car.physics.linear_velocity)
        
        # Only give dribble reward if car is moving toward target
        if car_vel_toward_target < 100:  # Car should be moving toward target
            return 0.0
        
        # Penalize if ball is moving much faster than car (indicates a hit, not dribble)
        ball_speed = np.linalg.norm(ball.linear_velocity)
        speed_diff = ball_speed - car_speed
        
        if speed_diff > self.max_ball_speed_diff:
            # Ball moving too fast relative to car - not dribbling
            return 0.0
        
        # Reward for ball and car moving in similar direction
        if car_speed > 50 and ball_speed > 50:
            car_vel_norm = car.physics.linear_velocity / car_speed
            ball_vel_norm = ball.linear_velocity / ball_speed
            vel_alignment = max(0, np.dot(car_vel_norm, ball_vel_norm))
        else:
            vel_alignment = 0.0
        
        # Taper off dribble reward as ball approaches wall (last target)
        # This encourages committing to the final hit rather than hovering
        dist_to_target = np.linalg.norm(ball.position - objective)
        # Full reward when far from target, tapers to 0.3x when within 500 units
        taper_start = 800.0
        taper_end = 300.0
        if dist_to_target < taper_start:
            taper = 0.3 + 0.7 * max(0, (dist_to_target - taper_end) / (taper_start - taper_end))
        else:
            taper = 1.0
        
        # Combined dribble reward:
        # - Ball is close to car
        # - Car is moving toward target
        # - Ball and car velocities are aligned
        # - Tapered near the wall
        dribble_reward = close_reward * vel_alignment * (car_vel_toward_target / CAR_MAX_SPEED) * taper
        
        return dribble_reward


class SetupBallSpeedPunishment(RewardFunction[AgentID, GameState, float]):
    """
    Punishes the ball moving too fast during setup phase.
    This discourages hitting the ball hard instead of dribbling it.
    """
    def __init__(self, speed_threshold: float = 1200.0):
        """
        :param speed_threshold: Ball speed above which punishment applies
        """
        self.speed_threshold = speed_threshold

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> float:
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        
        # Only active during setup
        if condition_data[SETUP_INDEX] < 0.5:
            return 0.0
        
        ball_speed = np.linalg.norm(state.ball.linear_velocity)
        
        if ball_speed > self.speed_threshold:
            # Punishment proportional to how much over threshold
            excess_speed = (ball_speed - self.speed_threshold) / BALL_MAX_SPEED
            return -excess_speed
        
        return 0.0


class SetupCompletionPunishment(RewardFunction[AgentID, GameState, float]):
    """
    NUCLEAR OPTION: Massive punishment if bot doesn't hit the next setup point
    within a time limit. Applies to ALL setup points, not just the last one.
    Forces the bot to keep progressing through setup.
    """
    def __init__(self, grace_seconds: float = 0.75, punishment_per_step: float = -5.0):
        """
        :param grace_seconds: Time allowed to hit each setup target before punishment starts
        :param punishment_per_step: Punishment applied every step after grace period
        """
        self.grace_seconds = grace_seconds
        self.punishment_per_step = punishment_per_step
        self.last_target_hit_tick = None
        self.last_known_target_idx = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_target_hit_tick = initial_state.tick_count
        self.last_known_target_idx = 0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> float:
        condition_data = shared_info["condition_data"]
        current_idx = shared_info["current_target_index"]
        
        # Check if current target is a setup point
        if current_idx >= len(condition_data):
            return 0.0
        
        in_setup = condition_data[current_idx][SETUP_INDEX] > 0.5
        
        # Only apply during setup phase
        if not in_setup:
            return 0.0
        
        # Check if target index changed (bot hit a target)
        if current_idx != self.last_known_target_idx:
            # Target was hit, reset timer
            self.last_target_hit_tick = state.tick_count
            self.last_known_target_idx = current_idx
            return 0.0
        
        # Check if grace period has expired
        time_since_last_hit = (state.tick_count - self.last_target_hit_tick) / TICKS_PER_SECOND
        
        if time_since_last_hit > self.grace_seconds:
            # Punishment time!
            return self.punishment_per_step
        
        return 0.0