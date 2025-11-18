import numpy as np
from typing import List, Dict, Any, Callable
import math
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_RADIUS, CAR_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BACK_NET_Y, SIDE_WALL_X, CEILING_Z

from math_utils import normalize

def height_sigmoid(height: float) -> float:
    return 0.5 * np.tanh((height - 900) / 250) + 0.5

class DistancePlayerToGround(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        height = min(state.cars[agent].physics.position[2], 1700)
        reward = 0.6 * np.tanh((height - 900) / 500) + 0.4
        return reward

class PlayerFallPunishment(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        z_fall = state.cars[agent].physics.linear_velocity[2]
        if z_fall > -50: z_fall = 0
        punish = 5.0 * z_fall / CAR_MAX_SPEED
        return punish

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
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        if state.cars[agent].physics.position[2] < 200.0 or state.ball.position[2] < 200.0:
            return 0.0

        # Compensate for inside of ball being unreachable (keep max reward at 1)
        dist = np.linalg.norm(state.cars[agent].physics.position - (state.ball.position + np.array([0, 0, -50]))) - BALL_RADIUS
        # the typical decay for regular gameplay is 0.5, but we use 12.0 to make it really strict for air dribbling.
        # makes it so it needs to be within 600 uu to get any measurable reward.
        dist_reward = np.exp(-12.0 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

        height_reward = height_sigmoid(state.cars[agent].physics.position[2])
        return dist_reward * height_reward



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

        if car.position[2] < 200.0 or ball.position[2] < 200.0:
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
        height_reward = height_sigmoid(state.cars[agent].physics.position[2])
        return vel_reward * height_reward

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
        thresh_floor = 300.0
        thresh_wall = 200.0
        close_to_wall = np.abs(state.ball.position[0]) > SIDE_WALL_X - thresh_wall
        close_to_wall = close_to_wall or np.abs(state.ball.position[1]) > BACK_WALL_Y - thresh_wall
        close_to_wall = close_to_wall or state.ball.position[2] > CEILING_Z - thresh_ceiling
        close_to_wall = close_to_wall or state.ball.position[2] < thresh_floor
        if close_to_wall:
            return -2.5*0
        height = state.ball.position[2]
        height_reward = height_sigmoid(height)
        return height_reward

class TouchReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward when agent touches ball.
    The more beneath the ball it hits, the higher the reward.
    Also gives an optional reward for accelerating the ball upward.
    """

    def __init__(self, acceleration_reward: float = 1.0):
        self.acceleration_reward = acceleration_reward
        self.prev_ball = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ball = initial_state.ball

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        hit_ball = 1. if state.cars[agent].ball_touches > 0 else 0.


        if state.cars[agent].physics.position[2] < 200.0 or state.ball.position[2] < 200.0:
            return 0.0

        to_ball = state.ball.position - state.cars[agent].physics.position
        to_ball = to_ball / np.linalg.norm(to_ball)
        vertical = to_ball[2]
        vertical = max(0.0, min(vertical, 0.7071))/0.7071 

        # measure how much upward velocity direction it gave the ball
        acceleration = (state.ball.linear_velocity - self.prev_ball.linear_velocity) / BALL_MAX_SPEED
        accel_dir_z = normalize(acceleration)[2]


        self.prev_ball = state.ball
        return hit_ball * vertical + self.acceleration_reward * accel_dir_z