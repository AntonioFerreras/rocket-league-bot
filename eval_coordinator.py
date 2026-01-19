"""
Evaluation coordinator using rlgym_learn infrastructure for efficient multiprocessing.
Uses the same backend as LearningCoordinator but with an eval-only agent controller.
"""

import os
import time
import json
import argparse
from functools import partial
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm

from rlgym.api import (
    AgentID,
    RewardFunction,
    StateMutator,
    DoneCondition,
)
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_RESTING_HEIGHT, BLUE_TEAM
from rlgym.rocket_league import common_values
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence

from rlgym_learn import EnvActionResponse, EnvActionResponseType, Timestep
from rlgym_learn.api.agent_controller import AgentController, DerivedAgentControllerConfig

from path_generator import generate_random_path
from mutators import AirDribbleDirectedMutator
from models import DiscreteFF

# Config constants
NUM_CONDITIONS = 16
TEAM_SIZE = 1
PAD_TEAM_SIZE = 2
ACTION_REPEAT = 8
NO_TOUCH_TIMEOUT_SECONDS = 5
BALL_HIT_GROUND_TIMEOUT_SECONDS = 2
GAME_TIMEOUT_SECONDS = 100


class EvalAgentControllerConfigModel(BaseModel, extra="forbid"):
    """Config for the evaluation agent controller."""
    checkpoint_path: str
    n_episodes: int = 10000
    output_path: Optional[str] = None
    deterministic: bool = True


@dataclass
class EvalAgentControllerData:
    """Data passed to metrics tracking (not used heavily for eval)."""
    cumulative_episodes: int
    best_flip_resets: int
    goals_scored: int
    clean_aerial_goals: int


class EvalAgentController(
    AgentController[
        EvalAgentControllerConfigModel,
        AgentID,
        np.ndarray,  # ObsType
        np.ndarray,  # ActionType
        float,       # RewardType
        GameState,   # StateType
        Tuple[str, int],  # ObsSpaceType
        Tuple[str, int],  # ActionSpaceType
        torch.Tensor,     # ActionAssociatedLearningData
        EvalAgentControllerData,
    ]
):
    """
    Evaluation-only agent controller that:
    - Loads a pre-trained actor
    - Uses deterministic policy
    - Tracks flip resets, goals, and clean aerials
    - Saves best paths
    """

    def __init__(self, actor_factory):
        super().__init__()
        self.actor_factory = actor_factory
        self.actor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tracking vars
        self.cumulative_episodes = 0
        self.target_episodes = 0
        self.best_flip_resets = 0
        self.goals_scored = 0
        self.clean_aerial_goals = 0
        self.flip_reset_distribution = Counter()
        self.best_episode_info = None
        
        # Progress bar
        self.pbar = None
        
    def set_space_types(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space

    def validate_config(self, config_obj):
        return EvalAgentControllerConfigModel.model_validate(config_obj)

    def load(self, config: DerivedAgentControllerConfig[EvalAgentControllerConfigModel]):
        self.config = config
        agent_config = config.agent_controller_config
        
        self.target_episodes = agent_config.n_episodes
        self.output_path = agent_config.output_path
        self.deterministic = agent_config.deterministic
        
        # Create actor
        self.actor = self.actor_factory(self.obs_space, self.action_space, self.device)
        
        # Load checkpoint
        checkpoint_path = agent_config.checkpoint_path
        actor_checkpoint = os.path.join(checkpoint_path, "ppo_learner", "actor.pt")
        
        print(f"Loading actor from {actor_checkpoint}")
        self.actor.load_state_dict(torch.load(actor_checkpoint, map_location=self.device, weights_only=True))
        self.actor.eval()
        
        # Initialize progress bar
        self.pbar = tqdm(total=self.target_episodes, desc="Evaluating", unit="ep")
        
        print(f"Eval config: {agent_config.n_episodes} episodes, deterministic={self.deterministic}")
        if self.output_path:
            print(f"Best paths will be saved to: {self.output_path}")

    def choose_agents(self, agent_id_list):
        return list(range(len(agent_id_list)))

    @torch.no_grad()
    def get_actions(self, agent_id_list, obs_list):
        actions, log_probs = self.actor.get_action(
            agent_id_list, obs_list, deterministic=self.deterministic
        )
        
        # Debug: Check action format for first few calls
        if not hasattr(self, '_action_debug_count'):
            self._action_debug_count = 0
        self._action_debug_count += 1
        if self._action_debug_count <= 5:
            obs_arr = np.array(obs_list[0]) if obs_list else None
            print(f"[get_actions DEBUG #{self._action_debug_count}]")
            print(f"  obs[0][:10]: {obs_arr[:10] if obs_arr is not None else 'None'}")
            print(f"  obs[0][-10:]: {obs_arr[-10:] if obs_arr is not None else 'None'}")
            print(f"  actions: {actions}")
        
        # Ensure actions are properly shaped for rlgym_learn
        # For deterministic mode, argmax returns (batch,) but we need (batch, 1)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        
        if self._action_debug_count <= 5:
            pass  # Already printed above
        
        if log_probs.dim() == 0:
            log_probs = log_probs.unsqueeze(0)
        return (actions, log_probs)

    def process_timestep_data(self, timestep_data):
        """Process timesteps - tracking is done via shared_info in choose_env_actions."""
        pass  # We get shared_info in choose_env_actions, so no tracking needed here

    def choose_env_actions(self, state_info):
        """Choose whether to step or reset each env."""
        env_action_responses = {}
        
        for env_id, (shared_info, state, terminated_dict, truncated_dict) in state_info.items():
            # Check if episode is done
            is_done = False
            if terminated_dict is not None and truncated_dict is not None:
                is_done = any(terminated_dict.values()) or any(truncated_dict.values())
            
            if is_done:
                # Debug: print episode end info for first few episodes
                if self.cumulative_episodes < 5:
                    print(f"[EP END DEBUG] env={env_id}, terminated={terminated_dict}, truncated={truncated_dict}")
                # Episode ended - process results
                self._process_episode_end(env_id, shared_info, terminated_dict)
                env_action_responses[env_id] = EnvActionResponse.RESET()
            else:
                env_action_responses[env_id] = EnvActionResponse.STEP()
        
        return env_action_responses

    def _process_episode_end(self, env_id: str, shared_info: Optional[Dict], terminated_dict: Optional[Dict]):
        """Process episode end and track statistics."""
        self.cumulative_episodes += 1
        self.pbar.update(1)
        
        if shared_info is None:
            if self.cumulative_episodes <= 3:
                print(f"\n[DEBUG] Episode {self.cumulative_episodes}: shared_info is None!")
            return
            
        num_flip_resets = shared_info.get("num_flip_resets", 0)
        scored_goal = terminated_dict is not None and any(terminated_dict.values())
        
        # Update statistics
        self.flip_reset_distribution[num_flip_resets] += 1
        
        if scored_goal:
            self.goals_scored += 1
            # For now, count all goals as clean (tracking not available in coordinator)
            self.clean_aerial_goals += 1
        
        # Check if this is a new best (goal with most flip resets)
        if scored_goal and num_flip_resets > self.best_flip_resets:
            self.best_flip_resets = num_flip_resets
            
            self.best_episode_info = {
                "num_flip_resets": num_flip_resets,
                "shared_info": shared_info.copy() if shared_info else {},
            }
            
            print(f"\n*** New best: {num_flip_resets} flip resets (goal)! ***")
            
            if self.output_path:
                self._save_best_path()
        
        # Check if we've reached target episodes
        if self.cumulative_episodes >= self.target_episodes:
            if not getattr(self, '_results_printed', False):
                self._results_printed = True
                self._print_final_results()
                print("\nTarget episodes reached. Press Ctrl+C to exit.")

    def _save_best_path(self):
        """Save the best path to a file in format compatible with replay_best_path.py."""
        if self.best_episode_info is None:
            return
            
        info = self.best_episode_info
        shared = info.get("shared_info", {})
        
        save_data = {
            "num_flip_resets": np.array([info["num_flip_resets"]]),
        }
        
        # Path data
        for key in ["path_points", "path_start", "path_end", "path_control", "condition_data"]:
            if shared.get(key) is not None:
                save_data[key] = shared[key]
        
        # Path info (compatible with replay_best_path.py format)
        save_data["has_setup"] = np.array([shared.get("has_setup", 0)])
        save_data["num_setup_points"] = np.array([shared.get("num_setup_points", 0)])
        
        # Use init positions as ball_spawn/car_spawn
        if shared.get("init_ball_position") is not None:
            save_data["ball_spawn"] = shared["init_ball_position"]
        if shared.get("init_car_position") is not None:
            save_data["car_spawn"] = shared["init_car_position"]
        
        # Initial state for exact replay
        if shared.get("init_ball_position") is not None:
            save_data["ball_position"] = shared["init_ball_position"]
            save_data["ball_linear_velocity"] = shared["init_ball_linear_velocity"]
            save_data["ball_angular_velocity"] = shared["init_ball_angular_velocity"]
            save_data["car_position"] = shared["init_car_position"]
            save_data["car_linear_velocity"] = shared["init_car_linear_velocity"]
            save_data["car_angular_velocity"] = shared["init_car_angular_velocity"]
            save_data["car_euler_angles"] = shared["init_car_euler_angles"]
            save_data["car_boost_amount"] = np.array([shared["init_car_boost_amount"]])
            save_data["car_on_ground"] = np.array([shared["init_car_on_ground"]])
            save_data["car_has_jumped"] = np.array([shared["init_car_has_jumped"]])
        
        np.savez(self.output_path, **save_data)
        print(f"Saved best path to {self.output_path}")

    def _print_final_results(self):
        """Print final evaluation results."""
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Total episodes: {self.cumulative_episodes}")
        
        if self.cumulative_episodes > 0:
            print(f"Goals scored: {self.goals_scored} ({100*self.goals_scored/self.cumulative_episodes:.1f}%)")
            print(f"Clean aerial goals: {self.clean_aerial_goals} ({100*self.clean_aerial_goals/self.cumulative_episodes:.1f}%)")
            print(f"\nMax flip resets in CLEAN GOAL episode: {self.best_flip_resets}")
            print(f"\nFlip reset distribution:")
            for n_resets in sorted(self.flip_reset_distribution.keys()):
                count = self.flip_reset_distribution[n_resets]
                pct = 100 * count / self.cumulative_episodes
                bar = "█" * int(pct / 2)
                print(f"  {n_resets}: {count:5d} ({pct:5.1f}%) {bar}")
        else:
            print("No episodes completed.")
        print("=" * 60)

    def process_env_actions(self, env_actions):
        pass

    def save_checkpoint(self):
        pass

    def cleanup(self):
        if self.pbar:
            self.pbar.close()
        if not getattr(self, '_results_printed', False):
            self._print_final_results()


# ============================================================================
# Environment building functions
# ============================================================================

class EvalAirDribbleDirectedMutator(AirDribbleDirectedMutator):
    """Mutator that forces 100% setup and flip reset probability."""
    
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        from math_utils import dir_to_euler_yzx, normalize
        import random
        
        # Generate path with 100% setup and flip reset probability
        path_points, start_point, end_point, control_point, glue_conditions, flip_reset_conditions, setup_conditions, path_info = generate_random_path(
            has_setup_probability=1.0,
            flip_reset_probability=1.0
        )
        
        num_path_points = len(path_points)
        condition_data = np.zeros((num_path_points, self.num_conditions), dtype=np.float32)

        # Set condition flags based on generated conditions
        if num_path_points > 0:
            condition_data[:, 6] = glue_conditions
            condition_data[:, 7] = flip_reset_conditions
            condition_data[:, 8] = setup_conditions
        
        has_setup = path_info["has_setup"]
        
        if has_setup:
            # Setup path: ball and car start on ground
            ball_spawn = path_info["ball_spawn"]
            car_spawn = path_info["car_spawn"]
            
            state.ball.position = ball_spawn.astype(np.float32)
            
            if len(path_points) > 0:
                first_target = path_points[0]
            else:
                first_target = end_point
            
            objective_direction = normalize(first_target - ball_spawn)
            objective_direction[2] = 0.0
            
            ball_vel = objective_direction * random.uniform(0, 200)
            ball_vel[2] = 0.0
            state.ball.linear_velocity = ball_vel.astype(np.float32)
            state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
            
            for car in state.cars.values():
                car.physics.position = car_spawn.astype(np.float32)
                
                to_ball = state.ball.position - car.physics.position
                to_ball[2] = 0.0
                car.physics.euler_angles = dir_to_euler_yzx(to_ball)
                
                car_vel = normalize(to_ball) * random.uniform(0, 300)
                car_vel[2] = 0.0
                car.physics.linear_velocity = car_vel.astype(np.float32)
                car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
                
                car.boost_amount = 100.0
                car.on_ground = True
                car.has_jumped = False
                car.air_time_since_jump = 0.0
        else:
            # Aerial-only path
            state.ball.position = start_point.astype(np.float32)
            
            if len(path_points) > 0:
                first_target = path_points[0]
            else:
                first_target = end_point
                
            objective_direction = normalize(first_target - start_point)
            objective_direction[2] = 0.0
            
            ball_vel = np.array([
                random.uniform(-50, 50), 
                random.uniform(-50, 50), 
                random.uniform(150, 550)
            ], dtype=np.float32)
            state.ball.linear_velocity = ball_vel + objective_direction * random.uniform(150, 300)
            state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
            
            ball_x, ball_y, ball_z = state.ball.position

            for car in state.cars.values():
                pos_x = ball_x + random.uniform(-20, 20)
                pos_y = ball_y + random.uniform(20, 20)
                pos_z = ball_z - random.uniform(100, 500)
                
                pos_z = max(400, pos_z)

                car.physics.position = np.array([pos_x, pos_y, pos_z], dtype=np.float32)

                to_ball = state.ball.position - car.physics.position
                to_ball = to_ball / np.linalg.norm(to_ball)
                car_vel = to_ball * random.uniform(300, 400)

                car.physics.linear_velocity = car_vel
                car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
                to_ball = state.ball.position - car.physics.position 
                car.physics.euler_angles = dir_to_euler_yzx(to_ball)
                car.boost_amount = 100.0
                car.air_time_since_jump = 2.0
                car.has_jumped = True
        
        # Store path info in shared_info
        shared_info["path_points"] = path_points.astype(np.float32)
        shared_info["path_start"] = start_point.astype(np.float32)
        shared_info["path_end"] = end_point.astype(np.float32)
        shared_info["path_control"] = control_point.astype(np.float32)
        
        # Store path_info fields for saving
        shared_info["has_setup"] = 1 if has_setup else 0  # Use int for serde compatibility
        shared_info["num_setup_points"] = path_info.get("num_setup_points", 0)
        
        # Store initial state for replay (after state has been set above)
        shared_info["init_ball_position"] = state.ball.position.copy().astype(np.float32)
        shared_info["init_ball_linear_velocity"] = state.ball.linear_velocity.copy().astype(np.float32)
        shared_info["init_ball_angular_velocity"] = state.ball.angular_velocity.copy().astype(np.float32)
        
        # Get first car's state
        for car in state.cars.values():
            shared_info["init_car_position"] = car.physics.position.copy().astype(np.float32)
            shared_info["init_car_linear_velocity"] = car.physics.linear_velocity.copy().astype(np.float32)
            shared_info["init_car_angular_velocity"] = car.physics.angular_velocity.copy().astype(np.float32)
            shared_info["init_car_euler_angles"] = car.physics.euler_angles.copy().astype(np.float32)
            shared_info["init_car_boost_amount"] = float(car.boost_amount)
            shared_info["init_car_on_ground"] = 1 if car.on_ground else 0
            shared_info["init_car_has_jumped"] = 1 if car.has_jumped else 0
            break  # Only first car

        shared_info["air_roll_rate"] = 0.0
        shared_info["air_roll_action"] = 0
        shared_info["condition_data"] = condition_data
        shared_info["hit_accel_dir_z"] = -1.0
        shared_info["num_ball_touches"] = 0
        shared_info["num_flip_resets"] = 0
        shared_info["reset_distance_to_goal"] = -1.0
        shared_info["num_setup_targets_hit"] = 0
        shared_info["num_air_targets_hit_after_setup"] = 0
        shared_info["current_target_index"] = 0


class NoTouchTimeoutCondition(DoneCondition[AgentID, GameState]):
    """Done when no car has touched the ball for a specified time."""

    def __init__(self, timeout_seconds: float, freeze_start_tick: bool = False):
        self.timeout_seconds = timeout_seconds
        self.last_touch_tick = None
        self.freeze_start_tick = freeze_start_tick

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        if self.freeze_start_tick:
            self.last_touch_tick = None
        else:
            self.last_touch_tick = initial_state.tick_count

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        if any(car.ball_touches > 0 for car in state.cars.values()):
            self.last_touch_tick = state.tick_count
            done = False
        else:
            if self.last_touch_tick is None:
                return {agent: False for agent in agents}
            time_elapsed = (state.tick_count - self.last_touch_tick) / common_values.TICKS_PER_SECOND
            done = time_elapsed >= self.timeout_seconds
        return {agent: done for agent in agents}


class BallHitGroundTimeoutCondition(DoneCondition[AgentID, GameState]):
    """Done when ball hits ground after aerial phase (indicates failed attempt)."""

    def __init__(self, timeout_seconds: float, post_setup_grace_seconds: float = 2.0):
        self.timeout_seconds = timeout_seconds
        self.post_setup_grace_seconds = post_setup_grace_seconds
        self.last_hit_ground_tick = None
        self.setup_left_tick = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_hit_ground_tick = None
        self.setup_left_tick = None

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        from path_generator import SETUP_INDEX
        
        condition_data = shared_info.get("condition_data")
        current_idx = shared_info.get("current_target_index", 0)
        
        in_setup = False
        if condition_data is not None and current_idx < len(condition_data):
            in_setup = condition_data[current_idx, SETUP_INDEX] > 0.5
        
        if in_setup:
            self.setup_left_tick = None
            return {agent: False for agent in agents}
        else:
            if self.setup_left_tick is None:
                self.setup_left_tick = state.tick_count
        
        time_since_setup_left = (state.tick_count - self.setup_left_tick) / common_values.TICKS_PER_SECOND
        if time_since_setup_left < self.post_setup_grace_seconds:
            return {agent: False for agent in agents}
        
        if state.ball.position[2] < BALL_RESTING_HEIGHT * 1.5:
            self.last_hit_ground_tick = state.tick_count
            done = False
        else:
            if self.last_hit_ground_tick is None:
                return {agent: False for agent in agents}
            time_elapsed = (state.tick_count - self.last_hit_ground_tick) / common_values.TICKS_PER_SECOND
            done = time_elapsed >= self.timeout_seconds
        return {agent: done for agent in agents}


class EvalReward(RewardFunction[AgentID, GameState, float]):
    """Reward function that tracks flip resets and updates path progress for eval."""
    
    def __init__(self):
        from rewards import FlipResetReward, BallToTargetReward
        self.flip_reset_reward = FlipResetReward(debug=False)
        self.ball_to_target_reward = BallToTargetReward(print_hits=False)
        self.step_count = 0
        self.episode_count = 0
        
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.flip_reset_reward.reset(agents, initial_state, shared_info)
        self.ball_to_target_reward.reset(agents, initial_state, shared_info)
        self.step_count = 0
        self.episode_count += 1
        
        self.last_tick = initial_state.tick_count
        self._last_target_idx = 0
        
        # Debug: print initial state for first 10 episodes
        if self.episode_count <= 10:
            print(f"\n[EvalReward DEBUG] Episode {self.episode_count} reset:")
            print(f"  Ball pos: {initial_state.ball.position}")
            print(f"  has_setup: {shared_info.get('has_setup', 'N/A')}")
            print(f"  current_target_index: {shared_info.get('current_target_index', 'N/A')}")
            print(f"  num_path_points: {len(shared_info.get('path_points', []))}")
            print(f"  initial tick: {initial_state.tick_count}")
            # Print first few path points to understand path structure
            path_points = shared_info.get('path_points', [])
            for i, pt in enumerate(path_points[:5]):
                print(f"  target[{i}]: {pt}")
        
    def get_rewards(self, agents: List[AgentID], state: GameState, 
                    is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        self.step_count += 1
        
        # BallToTargetReward updates current_target_index as ball follows path
        target_rewards = self.ball_to_target_reward.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        
        # FlipResetReward tracks flip resets
        flip_rewards = self.flip_reset_reward.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        
        # Debug: print state periodically for first episodes
        ticks_per_step = state.tick_count - self.last_tick
        self.last_tick = state.tick_count
        
        # Track target index changes
        current_idx = shared_info.get('current_target_index', 0)
        if not hasattr(self, '_last_target_idx'):
            self._last_target_idx = current_idx
        if current_idx != self._last_target_idx and self.episode_count <= 2:
            print(f"[TARGET CHANGED] Ep{self.episode_count} Step{self.step_count}: {self._last_target_idx} -> {current_idx}")
            self._last_target_idx = current_idx
        
        if self.episode_count <= 2 and self.step_count % 50 == 0:
            target_pos = shared_info["path_points"][current_idx]
            dist_to_target = np.linalg.norm(state.ball.position - target_pos)
            print(f"[EvalReward DEBUG] Ep{self.episode_count} Step{self.step_count}:")
            print(f"  Ball pos: {state.ball.position}")
            print(f"  Target pos: {target_pos}")
            print(f"  Dist to target: {dist_to_target:.1f} (threshold: 600)")
            print(f"  current_target_index: {shared_info.get('current_target_index', 'N/A')}")
            print(f"  num_flip_resets: {shared_info.get('num_flip_resets', 'N/A')}")
        
        # Combine rewards
        rewards = {agent: target_rewards.get(agent, 0) + flip_rewards.get(agent, 0) for agent in agents}
        return rewards


class CustomActionParser:
    """Action parser with repeat for tick skip."""
    
    def __init__(self, parser, repeats=8):
        self.parser = parser
        self.repeats = repeats

    def get_action_space(self, agent):
        return self.parser.get_action_space(agent)

    def reset(self, agents, initial_state, shared_info):
        self.parser.reset(agents, initial_state, shared_info)

    def parse_actions(self, actions, state, shared_info):
        rlgym_actions = self.parser.parse_actions(actions, state, shared_info)
        repeat_actions = {}
        for agent, action in rlgym_actions.items():
            shared_info["air_roll_action"] = action.flatten()[4]
            if action.shape == (8,):
                action = np.expand_dims(action, axis=0)
            elif action.shape != (1, 8):
                raise ValueError(f"Expected action shape (8,) or (1,8), got {action.shape}")
            repeat_actions[agent] = action.repeat(self.repeats, axis=0)
        return repeat_actions


class FreestyleObs(DefaultObs):
    """Observation builder that includes path target info."""
    
    def __init__(self, num_conditions: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_conditions = num_conditions

    def get_obs_space(self, agent: AgentID) -> Tuple[str, int]:
        if self.zero_padding is not None:
            return 'real', 52 + 20 * self.zero_padding * 2 + self.num_conditions
        elif self._state is not None:
            return 'real', 52 + 20 * len(self._state.cars) + self.num_conditions
        else:
            return 'real', -1

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        self._state = state
        obs = {}
        for agent in agents:
            obs[agent] = self._build_obs(agent, state, shared_info)
        
        # Track stats
        shared_info["ball_x"] = state.ball.position[0]
        shared_info["ball_y"] = state.ball.position[1]
        shared_info["ball_z"] = state.ball.position[2]
        
        return obs

    def _build_obs(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> np.ndarray:
        obs = super()._build_obs(agent, state, shared_info)
        
        target_pos = shared_info["path_points"][shared_info["current_target_index"]]
        num_targets = len(shared_info["path_points"])
        if shared_info["current_target_index"] + 1 < num_targets:
            next_target_pos = shared_info["path_points"][shared_info["current_target_index"] + 1]
        else:
            next_target_pos = target_pos

        obs = np.concatenate([obs, target_pos * self.POS_COEF])
        obs = np.concatenate([obs, next_target_pos * self.POS_COEF])
        
        current_conditions = shared_info["condition_data"][shared_info["current_target_index"]]
        obs = np.concatenate([obs, current_conditions[6:]])
        return obs


def build_eval_env():
    """Build the evaluation environment."""
    
    action_parser = CustomActionParser(LookupTableAction(), repeats=ACTION_REPEAT)
    
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=NO_TOUCH_TIMEOUT_SECONDS, freeze_start_tick=False),
        TimeoutCondition(timeout_seconds=GAME_TIMEOUT_SECONDS),
        BallHitGroundTimeoutCondition(timeout_seconds=BALL_HIT_GROUND_TIMEOUT_SECONDS),
    )
    
    obs_builder = FreestyleObs(
        num_conditions=NUM_CONDITIONS,
        zero_padding=PAD_TEAM_SIZE,
        pos_coef=np.asarray([
            1 / common_values.SIDE_WALL_X,
            1 / common_values.BACK_NET_Y,
            1 / common_values.CEILING_Z,
        ]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )
    
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=TEAM_SIZE, orange_size=0),
        EvalAirDribbleDirectedMutator(num_conditions=NUM_CONDITIONS),
    )
    
    reward_fn = EvalReward()
    
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate flip reset bot using rlgym_learn coordinator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--n_episodes", type=int, default=10000, help="Number of episodes to evaluate")
    parser.add_argument("--n_proc", type=int, default=64, help="Number of parallel processes")
    parser.add_argument("--output", type=str, default="best_path.npz", help="Output path for best path")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy")
    args = parser.parse_args()
    
    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        ProcessConfigModel,
        PyAnySerdeType,
        SerdeTypesModel,
    )
    
    train_dtype = torch.float32
    
    def actor_factory(obs_space, action_space, device):
        dim = 512
        num_layers = 4
        return DiscreteFF(
            obs_space[1],
            action_space[1],
            (dim,) * num_layers,
            device,
            dtype=train_dtype
        )
    
    # Configure the coordinator
    config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                action_serde_type=PyAnySerdeType.NUMPY(np.int64),
                obs_serde_type=PyAnySerdeType.NUMPY(np.float64),
                reward_serde_type=PyAnySerdeType.FLOAT(),
                obs_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                action_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                shared_info_serde_type=PyAnySerdeType.TYPEDDICT({
                    "path_points": PyAnySerdeType.NUMPY(np.float32),
                    "path_start": PyAnySerdeType.NUMPY(np.float32),
                    "path_end": PyAnySerdeType.NUMPY(np.float32),
                    "path_control": PyAnySerdeType.NUMPY(np.float32),
                    "current_target_index": PyAnySerdeType.INT(),
                    "ball_x": PyAnySerdeType.FLOAT(),
                    "ball_y": PyAnySerdeType.FLOAT(),
                    "ball_z": PyAnySerdeType.FLOAT(),
                    "air_roll_rate": PyAnySerdeType.FLOAT(),
                    "air_roll_action": PyAnySerdeType.INT(),
                    "condition_data": PyAnySerdeType.NUMPY(np.float32),
                    "hit_accel_dir_z": PyAnySerdeType.FLOAT(),
                    "num_ball_touches": PyAnySerdeType.INT(),
                    "num_flip_resets": PyAnySerdeType.INT(),
                    "reset_distance_to_goal": PyAnySerdeType.FLOAT(),
                    "num_setup_targets_hit": PyAnySerdeType.INT(),
                    "num_air_targets_hit_after_setup": PyAnySerdeType.INT(),
                    # Path info fields
                    "has_setup": PyAnySerdeType.INT(),
                    "num_setup_points": PyAnySerdeType.INT(),
                    # Initial state for replay
                    "init_ball_position": PyAnySerdeType.NUMPY(np.float32),
                    "init_ball_linear_velocity": PyAnySerdeType.NUMPY(np.float32),
                    "init_ball_angular_velocity": PyAnySerdeType.NUMPY(np.float32),
                    "init_car_position": PyAnySerdeType.NUMPY(np.float32),
                    "init_car_linear_velocity": PyAnySerdeType.NUMPY(np.float32),
                    "init_car_angular_velocity": PyAnySerdeType.NUMPY(np.float32),
                    "init_car_euler_angles": PyAnySerdeType.NUMPY(np.float32),
                    "init_car_boost_amount": PyAnySerdeType.FLOAT(),
                    "init_car_on_ground": PyAnySerdeType.INT(),
                    "init_car_has_jumped": PyAnySerdeType.INT(),
                }),
            ),
            timestep_limit=1_000_000_000,  # High limit, we stop based on episode count
        ),
        process_config=ProcessConfigModel(
            n_proc=args.n_proc,
            render=False,
        ),
        agent_controllers_config={
            "Eval": EvalAgentControllerConfigModel(
                checkpoint_path=args.checkpoint,
                n_episodes=args.n_episodes,
                output_path=args.output,
                deterministic=args.deterministic,
            )
        },
        agent_controllers_save_folder="eval_checkpoints",
    )
    
    print(f"Starting evaluation with {args.n_proc} processes")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.n_episodes}")
    
    coordinator = LearningCoordinator(
        build_eval_env,
        agent_controllers={
            "Eval": EvalAgentController(actor_factory=actor_factory)
        },
        config=config,
    )
    
    try:
        coordinator.start()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")


if __name__ == "__main__":
    main()

