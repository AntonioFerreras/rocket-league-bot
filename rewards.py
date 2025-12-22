import numpy as np
from typing import List, Dict, Any, Callable
import math
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_RADIUS, CAR_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BACK_NET_Y, SIDE_WALL_X, CEILING_Z, CAR_MAX_ANG_VEL, TICKS_PER_SECOND

from math_utils import normalize, cosine_similarity

from path_generator import GORILLA_GLUE_INDEX, FLIP_RESET_INDEX

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
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        if state.cars[agent].physics.position[2] < 60.0:
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

        if state.cars[agent].physics.position[2] < 100.0 or state.ball.position[2] < 150.0:
            return 0.0

        # Compensate for inside of ball being unreachable (keep max reward at 1)

        if condition_data[GORILLA_GLUE_INDEX] < 0.5:
            return 0.0

        dist = np.linalg.norm(state.cars[agent].physics.position - (state.ball.position)) - BALL_RADIUS
        dist_reward = np.exp(-100.0 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

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
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState):
        ball = state.ball
        car = state.cars[agent].physics

        if car.position[2] < 100.0 or ball.position[2] < 150.0:
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
    def __init__(self, target_distance=500, print_hits=False):
        super().__init__()
        self.target_distance = target_distance
        self.print_hits = print_hits

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        shared_info["current_target_index"] = 0

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state, shared_info) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> float:
        objective = shared_info["path_points"][shared_info["current_target_index"]]
        dir_target = normalize(objective - state.ball.position)
        dot_product = np.dot(dir_target, state.ball.linear_velocity/BALL_MAX_SPEED)
        dist = np.linalg.norm(state.ball.position - objective)
        if dist < self.target_distance:
            if self.print_hits:
                print(f"Hit target {shared_info['current_target_index']}")
            shared_info["current_target_index"] = min(shared_info["current_target_index"] + 1, len(shared_info["path_points"]) - 1)
            hit_target_reward = 1.0
        else:
            if self.print_hits:
                print(f"Not hit target {shared_info['current_target_index']}")
            hit_target_reward = 0.0
        reward = hit_target_reward + dot_product*5.0
        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        if condition_data[GORILLA_GLUE_INDEX] < 0.5 and condition_data[FLIP_RESET_INDEX] < 0.5:
            roll_rate = AirRollReward.get_air_roll_rate(state, agent)
            reward *= abs(roll_rate)
        if condition_data[FLIP_RESET_INDEX] > 0.5:
            reward *= 0.4
        return reward


class ForwardBiasReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward for moving in the direction the car is facing.
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(
        self, agent: AgentID, state: GameState
    ) -> float:
        return state.cars[agent].physics.forward.dot(normalize(state.cars[agent].physics.linear_velocity))

class ZoneReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a punishment when agent is close to a wall, ceiling, or ground.
    Gives reward when in a good height range.
    To prevent the agent from driving on walls to avoid low height punishment.
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(
        self, agent: AgentID, state: GameState
    ) -> float:
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
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(
        self, agent: AgentID, state: GameState
    ) -> float:
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
            reward = state.cars[agent].ball_touches * vertical
        elif condition_data[FLIP_RESET_INDEX] < 0.5:
            
            reward = accel_dir_z*accel_dir_z * hit_ball * 0.25
            roll_rate = AirRollReward.get_air_roll_rate(state, agent)
            reward *= abs(roll_rate)
            reward -= hit_ball*state.cars[agent].ball_touches*3.5
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
    def __init__(self, obtain_flip_weight: float = 6.0, hit_ball_weight: float = 6.0, down_facing_ball_weight: float = 0.8):
        self.obtain_flip_weight = obtain_flip_weight
        self.hit_ball_weight = hit_ball_weight
        self.down_facing_ball_weight = down_facing_ball_weight

        self.prev_state = None
        self.has_reset = None
        self.has_flipped = None
        self.down_facing_juice = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_state = initial_state
        self.has_reset = set()
        self.has_flipped = set()
        self.down_facing_juice = self.obtain_flip_weight
        
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {k: 0.0 for k in agents}

        condition_data = shared_info["condition_data"][shared_info["current_target_index"]]
        if condition_data[FLIP_RESET_INDEX] < 0.5:
            return rewards

        for agent in agents:
            car = state.cars[agent]
            down = -car.physics.up
            car_ball = state.ball.position - car.physics.position
            cossim_down_ball = max(0, cosine_similarity(down, car_ball))
            if car.ball_touches > 0 and car.has_flip and not self.prev_state.cars[agent].has_flip:
                if cossim_down_ball > 0.5 ** 0.5:  # 45 degrees
                    self.has_reset.add(agent)
                    rewards[agent] = 0.0
                    if state.cars[agent].physics.position[2] > 80.0:
                        rewards[agent] += self.obtain_flip_weight
                    shared_info["num_flip_resets"] += 1
            elif car.on_ground:
                self.has_reset.discard(agent)
                self.has_flipped.discard(agent)
            elif car.is_flipping and agent in self.has_reset:
                self.has_reset.remove(agent)
                self.has_flipped.add(agent)
                self.down_facing_juice = self.obtain_flip_weight
            if car.ball_touches > 0 and agent in self.has_flipped:
                self.has_flipped.remove(agent)
                rewards[agent] = self.hit_ball_weight
            
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
                pointing_in_motion_direction = np.dot(normalize(state.cars[agent].physics.forward[:2]), normalize(state.cars[agent].physics.linear_velocity[:2])) > 0.766044443
                close_reward = self.down_facing_ball_weight * (cossim_down_ball*2.0 - 1.0) * (0.5 if not pointing_in_motion_direction else 1.0) # np.exp(-60.0 * curr_dist_to_ball / CAR_MAX_SPEED)  # * max(0.0, (prev_dist_to_ball - curr_dist_to_ball)) / CAR_MAX_SPEED
                if decrease_rate > 30.0 and close_reward > 0.0:
                    close_reward *= 2.0
                
                rewards[agent] += close_reward
                self.down_facing_juice -= close_reward
        self.prev_state = state
        return rewards