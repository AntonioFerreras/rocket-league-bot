import numpy as np
from typing import List, Dict, Any

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_RADIUS, CAR_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BACK_NET_Y

class DistancePlayerToBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        dist = np.linalg.norm(state.cars[agent].physics.position - state.ball.position) - BALL_RADIUS
        return np.exp(-0.5 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

class DistanceBallToGoalReward(RewardFunction[AgentID, GameState, float]):
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
        return np.exp(-0.5 * dist / BALL_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196


class TouchReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward when agent touches ball.
    The more beneath the ball it hits, the higher the reward.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        hit_ball = 1. if state.cars[agent].ball_touches > 0 else 0.
        to_ball = state.ball.position - state.cars[agent].physics.position
        to_ball = to_ball / np.linalg.norm(to_ball)
        vertical = to_ball[2]
        vertical = max(0.25, min(vertical, 0.9))
        return hit_ball * vertical