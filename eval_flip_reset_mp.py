"""
Multiprocess evaluation script to find the path with the most flip resets.
Uses 100% setup and flip reset probability paths with deterministic policy.
Runs multiple environments in parallel for faster evaluation.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import random
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Queue, Process
from dataclasses import dataclass
from tqdm import tqdm

# Config
num_conditions = 16
spawn_opponents = False
team_size = 1
pad_team_size = 2
blue_team_size = team_size
orange_team_size = team_size if spawn_opponents else 0
action_repeat = 8
no_touch_timeout_seconds = 5
ball_hit_ground_timeout_seconds = 2
game_timeout_seconds = 100


@dataclass
class EpisodeResult:
    """Result from a single episode evaluation."""
    num_flip_resets: int
    scored_goal: bool
    clean_aerial: bool
    num_ball_touches: int
    num_setup_targets_hit: int
    num_air_targets_hit_after_setup: int
    # Path data for saving (only populated if this is a candidate for best)
    path_data: Optional[Dict] = None
    initial_state: Optional[Dict] = None


def build_eval_env(has_setup_probability=1.0, flip_reset_probability=1.0):
    """Build environment with custom setup/flip_reset probabilities."""
    from rlgym.rocket_league.common_values import BALL_RESTING_HEIGHT
    from rlgym.api import RLGym, StateMutator, DoneCondition, AgentID
    from rlgym.rocket_league.api import GameState
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction
    from rlgym.rocket_league.done_conditions import (
        AnyCondition,
        GoalCondition,
        TimeoutCondition,
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        MutatorSequence,
    )
    from math_utils import dir_to_euler_yzx, normalize
    from path_generator import generate_random_path

    class NoTouchTimeoutCondition(DoneCondition[AgentID, GameState]):
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

    from rlgym.api import ActionParser, ActionType, StateType, ActionSpaceType
    
    class CustomActionParser(ActionParser[AgentID, ActionType, np.ndarray, StateType, ActionSpaceType]):
        def __init__(self, parser, repeats=8):
            super().__init__()
            self.parser = parser
            self.repeats = repeats

        def get_action_space(self, agent: AgentID):
            return self.parser.get_action_space(agent)

        def reset(self, agents: List[AgentID], initial_state: StateType, shared_info: Dict[str, Any]) -> None:
            self.parser.reset(agents, initial_state, shared_info)

        def parse_actions(self, actions: Dict[AgentID, ActionType], state: StateType, shared_info: Dict[str, Any]):
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

    class EvalAirDribbleDirectedMutator(StateMutator[GameState]):
        def __init__(self, num_conditions: int, has_setup_probability: float, flip_reset_probability: float):
            self.num_conditions = num_conditions
            self.has_setup_probability = has_setup_probability
            self.flip_reset_probability = flip_reset_probability

        def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
            state.config.boost_consumption = 0.001
            
            path_points, start_point, end_point, control_point, glue_conditions, flip_reset_conditions, setup_conditions, path_info = generate_random_path(
                step_distance=1000,
                has_setup_probability=self.has_setup_probability,
                flip_reset_probability=self.flip_reset_probability
            )

            num_path_points = len(path_points)
            condition_data = np.zeros((num_path_points, self.num_conditions), dtype=np.float32)

            if num_path_points > 0:
                condition_data[:, 6] = glue_conditions
                condition_data[:, 7] = flip_reset_conditions
                condition_data[:, 8] = setup_conditions
            
            has_setup = path_info["has_setup"]
            
            if has_setup:
                ball_spawn = path_info["ball_spawn"]
                car_spawn = path_info["car_spawn"]
                
                state.ball.position = ball_spawn.astype(np.float32)
                
                first_target = path_points[0] if len(path_points) > 0 else end_point
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
                state.ball.position = start_point.astype(np.float32)
                
                first_target = path_points[0] if len(path_points) > 0 else end_point
                objective_direction = normalize(first_target - start_point)
                objective_direction[2] = 0.0
                
                ball_vel = np.array([
                    random.uniform(-50, 50), 
                    random.uniform(-50, 50), 
                    random.uniform(150, 550)
                ], dtype=np.float32)
                state.ball.linear_velocity = ball_vel + objective_direction * random.uniform(150, 300)
                state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
                
                FLOOR_MARGIN = 400
                CEILING_MARGIN = 450
                WALL_MARGIN = 100
                spawn_min_x = -common_values.SIDE_WALL_X + WALL_MARGIN
                spawn_max_x = common_values.SIDE_WALL_X - WALL_MARGIN
                spawn_min_y = -common_values.BACK_WALL_Y + WALL_MARGIN
                spawn_max_y = common_values.BACK_WALL_Y - WALL_MARGIN
                spawn_min_z = FLOOR_MARGIN
                spawn_max_z = common_values.CEILING_Z - CEILING_MARGIN
                
                car_min_height_under_ball = 100 
                car_max_height_under_ball = 500
                car_x_radius = 20
                car_y_min = 20
                car_y_max = 20
                car_speed_min = 300
                car_speed_max = 400
                
                ball_x, ball_y, ball_z = state.ball.position

                for car in state.cars.values():
                    pos_x = ball_x + random.uniform(-car_x_radius, car_x_radius)
                    pos_y = ball_y + random.uniform(car_y_min, car_y_max)
                    pos_z = ball_z - random.uniform(car_min_height_under_ball, car_max_height_under_ball)

                    pos_x = max(spawn_min_x, min(pos_x, spawn_max_x))
                    pos_y = max(spawn_min_y, min(pos_y, spawn_max_y))
                    pos_z = max(spawn_min_z, min(pos_z, spawn_max_z))

                    car.physics.position = np.array([pos_x, pos_y, pos_z], dtype=np.float32)

                    to_ball = state.ball.position - car.physics.position
                    to_ball = to_ball / np.linalg.norm(to_ball)
                    car_vel = to_ball * random.uniform(car_speed_min, car_speed_max)

                    car.physics.linear_velocity = car_vel
                    car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
                    to_ball = state.ball.position - car.physics.position 
                    car.physics.euler_angles = dir_to_euler_yzx(to_ball)
                    car.boost_amount = 100.0
                    car.air_time_since_jump = 2.0
                    car.has_jumped = True
            
            shared_info["path_points"] = path_points.astype(np.float32)
            shared_info["path_start"] = start_point.astype(np.float32)
            shared_info["path_end"] = end_point.astype(np.float32)
            shared_info["path_control"] = control_point.astype(np.float32)
            shared_info["path_info"] = path_info
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

    class FreestyleObs(DefaultObs):
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

    from rlgym.api import RewardFunction
    from rewards import FlipResetReward, BallToTargetReward

    class EvalReward(RewardFunction[AgentID, GameState, float]):
        def __init__(self):
            self.flip_reset_reward = FlipResetReward(debug=False, target_hit_timeout=10.0)
            self.ball_to_target_reward = BallToTargetReward(print_hits=False)
        
        def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
            self.flip_reset_reward.reset(agents, initial_state, shared_info)
            self.ball_to_target_reward.reset(agents, initial_state, shared_info)

        def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated, is_truncated, shared_info: Dict[str, Any]):
            self.flip_reset_reward.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            self.ball_to_target_reward.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            return {agent: 0.0 for agent in agents}

    action_parser = CustomActionParser(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds, freeze_start_tick=False),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
        BallHitGroundTimeoutCondition(timeout_seconds=ball_hit_ground_timeout_seconds),
    )

    obs_builder = FreestyleObs(
        num_conditions=num_conditions,
        zero_padding=pad_team_size,
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
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        EvalAirDribbleDirectedMutator(
            num_conditions=num_conditions,
            has_setup_probability=has_setup_probability,
            flip_reset_probability=flip_reset_probability
        ),
    )
    
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=EvalReward(),
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=None,
    )


def worker_process(worker_id: int, checkpoint_path: str, num_episodes: int, 
                   result_queue: Queue, seed: int, min_flip_resets_to_save: int = 0):
    """Worker process that runs episodes and sends results back."""
    from path_generator import SETUP_INDEX
    from rlgym.rocket_league.common_values import BALL_RESTING_HEIGHT
    from models import DiscreteFF
    
    # Set unique seed for this worker
    worker_seed = seed + worker_id * 10000
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build environment
    env = build_eval_env(has_setup_probability=1.0, flip_reset_probability=1.0)
    
    # Load actor
    obs_space = env.observation_space("blue-0")
    action_space = env.action_space("blue-0")
    
    dim = 512
    num_layers = 4
    actor = DiscreteFF(
        obs_space[1],
        action_space[1],
        (dim,) * num_layers,
        device,
        dtype=torch.float32
    )
    
    actor_path = os.path.join(checkpoint_path, "ppo_learner", "actor.pt")
    actor.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True))
    actor.eval()
    
    MIN_AIRBORNE_HEIGHT = 400
    GROUND_THRESHOLD = BALL_RESTING_HEIGHT * 1.5
    
    for ep in range(num_episodes):
        obs_dict = env.reset()
        done = False
        
        # Capture initial state
        shared_info = env.shared_info
        state = env.state
        car = list(state.cars.values())[0]
        
        initial_state_data = {
            "ball_position": state.ball.position.copy(),
            "ball_linear_velocity": state.ball.linear_velocity.copy(),
            "ball_angular_velocity": state.ball.angular_velocity.copy(),
            "car_position": car.physics.position.copy(),
            "car_linear_velocity": car.physics.linear_velocity.copy(),
            "car_angular_velocity": car.physics.angular_velocity.copy(),
            "car_euler_angles": car.physics.euler_angles.copy(),
            "car_boost_amount": np.array([car.boost_amount]),
            "car_on_ground": np.array([car.on_ground]),
            "car_has_jumped": np.array([car.has_jumped]),
            "car_air_time_since_jump": np.array([car.air_time_since_jump]),
        }
        
        # Deep copy path_info to avoid reference issues with numpy arrays
        raw_path_info = shared_info.get("path_info", {})
        path_info_copy = {
            "has_setup": raw_path_info.get("has_setup", False),
            "ball_spawn": raw_path_info.get("ball_spawn").copy() if raw_path_info.get("ball_spawn") is not None else None,
            "car_spawn": raw_path_info.get("car_spawn").copy() if raw_path_info.get("car_spawn") is not None else None,
            "num_setup_points": raw_path_info.get("num_setup_points", 0),
        }
        
        path_data = {
            "path_points": shared_info.get("path_points", None).copy() if shared_info.get("path_points") is not None else None,
            "path_start": shared_info.get("path_start", None).copy() if shared_info.get("path_start") is not None else None,
            "path_end": shared_info.get("path_end", None).copy() if shared_info.get("path_end") is not None else None,
            "path_control": shared_info.get("path_control", None).copy() if shared_info.get("path_control") is not None else None,
            "condition_data": shared_info.get("condition_data", None).copy() if shared_info.get("condition_data") is not None else None,
            "path_info": path_info_copy,
        }
        
        # Track clean aerial
        ball_hit_ground_after_setup = False
        left_setup_phase = False
        ball_was_airborne = False
        
        while not done:
            agent_ids = list(obs_dict.keys())
            obs_list = [obs_dict[agent_id] for agent_id in agent_ids]
            
            with torch.no_grad():
                actions, _ = actor.get_action(agent_ids, obs_list, deterministic=True)
            
            action_dict = {agent_id: np.array([actions[i]]) for i, agent_id in enumerate(agent_ids)}
            obs_dict, rewards, terminated, truncated = env.step(action_dict)
            
            # Track setup phase and ball height
            shared_info = env.shared_info
            condition_data = shared_info.get("condition_data")
            current_idx = shared_info.get("current_target_index", 0)
            in_setup = False
            if condition_data is not None and current_idx < len(condition_data):
                in_setup = condition_data[current_idx, SETUP_INDEX] > 0.5
            
            if not in_setup:
                left_setup_phase = True
            
            if left_setup_phase:
                ball_z = env.state.ball.position[2]
                if not ball_was_airborne:
                    if ball_z > MIN_AIRBORNE_HEIGHT:
                        ball_was_airborne = True
                else:
                    if ball_z < GROUND_THRESHOLD:
                        ball_hit_ground_after_setup = True
            
            done = any(terminated.values()) or any(truncated.values())
        
        # Collect result
        num_flip_resets = shared_info.get("num_flip_resets", 0)
        scored_goal = any(terminated.values())
        clean_aerial = not ball_hit_ground_after_setup
        
        # Only include path data if this is a good candidate
        include_path_data = (scored_goal and clean_aerial and num_flip_resets >= min_flip_resets_to_save)
        
        result = EpisodeResult(
            num_flip_resets=num_flip_resets,
            scored_goal=scored_goal,
            clean_aerial=clean_aerial,
            num_ball_touches=shared_info.get("num_ball_touches", 0),
            num_setup_targets_hit=shared_info.get("num_setup_targets_hit", 0),
            num_air_targets_hit_after_setup=shared_info.get("num_air_targets_hit_after_setup", 0),
            path_data=path_data if include_path_data else None,
            initial_state=initial_state_data if include_path_data else None,
        )
        
        result_queue.put((worker_id, result))
    
    env.close()
    result_queue.put((worker_id, None))  # Signal worker is done


def save_best_path(path_data: Dict, initial_state: Dict, num_flip_resets: int, output_path: str):
    """Save the best path data to a file."""
    path_info = path_data.get("path_info", {})
    
    np.savez(
        output_path,
        path_points=path_data["path_points"],
        path_start=path_data["path_start"],
        path_end=path_data["path_end"],
        path_control=path_data["path_control"],
        condition_data=path_data["condition_data"],
        has_setup=np.array([path_info.get("has_setup", False)]),
        ball_spawn=path_info.get("ball_spawn") if path_info.get("ball_spawn") is not None else np.array([]),
        car_spawn=path_info.get("car_spawn") if path_info.get("car_spawn") is not None else np.array([]),
        num_setup_points=np.array([path_info.get("num_setup_points", 0)]),
        expected_flip_resets=np.array([num_flip_resets]),  # Save for verification during replay
        ball_position=initial_state["ball_position"],
        ball_linear_velocity=initial_state["ball_linear_velocity"],
        ball_angular_velocity=initial_state["ball_angular_velocity"],
        car_position=initial_state["car_position"],
        car_linear_velocity=initial_state["car_linear_velocity"],
        car_angular_velocity=initial_state["car_angular_velocity"],
        car_euler_angles=initial_state["car_euler_angles"],
        car_boost_amount=initial_state["car_boost_amount"],
        car_on_ground=initial_state["car_on_ground"],
        car_has_jumped=initial_state["car_has_jumped"],
        car_air_time_since_jump=initial_state["car_air_time_since_jump"],
    )


def main():
    parser = argparse.ArgumentParser(description="Multiprocess evaluation for flip resets")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint folder containing ppo_learner/actor.pt")
    parser.add_argument("--num_episodes", type=int, default=10000,
                        help="Total number of episodes to run")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel worker processes")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random based on current time)")
    parser.add_argument("--output", type=str, default="best_path.npz",
                        help="Output file for best path data")
    args = parser.parse_args()
    
    # Use time-based seed if not specified
    if args.seed is None:
        args.seed = int(time.time() * 1000) % (2**32)
    print(f"Using seed: {args.seed}")
    
    print(f"Starting multiprocess evaluation with {args.num_workers} workers")
    print(f"Total episodes: {args.num_episodes}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Only paths that end in GOAL + CLEAN AERIAL will be saved as best\n")
    
    # Distribute episodes across workers
    episodes_per_worker = args.num_episodes // args.num_workers
    remainder = args.num_episodes % args.num_workers
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Start workers
    workers = []
    for i in range(args.num_workers):
        num_eps = episodes_per_worker + (1 if i < remainder else 0)
        p = Process(target=worker_process, args=(
            i, args.checkpoint, num_eps, result_queue, args.seed, 0
        ))
        p.start()
        workers.append(p)
    
    # Collect results
    best_flip_resets = 0
    best_result = None
    flip_reset_counts = []
    goals_scored = 0
    clean_goals = 0
    workers_done = 0
    
    pbar = tqdm(total=args.num_episodes, desc="Evaluating")
    
    while workers_done < args.num_workers:
        worker_id, result = result_queue.get()
        
        if result is None:
            workers_done += 1
            continue
        
        pbar.update(1)
        flip_reset_counts.append(result.num_flip_resets)
        
        if result.scored_goal:
            goals_scored += 1
        
        if result.scored_goal and result.clean_aerial:
            clean_goals += 1
            
            # Check if this is best
            if result.num_flip_resets > best_flip_resets:
                best_flip_resets = result.num_flip_resets
                best_result = result
                tqdm.write(f"[Worker {worker_id}] New best: {result.num_flip_resets} flip resets (GOAL + CLEAN AERIAL)!")
                
                # Save immediately
                if result.path_data is not None and result.initial_state is not None:
                    save_best_path(result.path_data, result.initial_state, result.num_flip_resets, args.output)
                    tqdm.write(f"  Saved best path to: {args.output}")
    
    pbar.close()
    
    # Wait for all workers to finish
    for p in workers:
        p.join()
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total episodes: {args.num_episodes}")
    print(f"Workers used: {args.num_workers}")
    print(f"Goals scored: {goals_scored} ({100*goals_scored/args.num_episodes:.1f}%)")
    print(f"Clean aerial goals (no ground touch): {clean_goals} ({100*clean_goals/args.num_episodes:.1f}%)")
    print(f"Episodes with at least 1 flip reset: {sum(1 for x in flip_reset_counts if x > 0)}")
    print(f"Average flip resets per episode: {np.mean(flip_reset_counts):.2f}")
    print(f"Max flip resets in CLEAN GOAL episode: {best_flip_resets}")
    
    if best_result:
        print("\n" + "-" * 60)
        print("BEST EPISODE INFO (goal + clean aerial):")
        print("-" * 60)
        print(f"  Flip resets: {best_result.num_flip_resets}")
        print(f"  Ball touches: {best_result.num_ball_touches}")
        print(f"  Setup targets hit: {best_result.num_setup_targets_hit}")
        print(f"  Air targets hit after setup: {best_result.num_air_targets_hit_after_setup}")
    else:
        print("\n" + "-" * 60)
        print("WARNING: No episodes met criteria (goal + clean aerial)! No best path saved.")
        print("-" * 60)
    
    # Distribution
    print("\n" + "-" * 60)
    print("FLIP RESET DISTRIBUTION:")
    print("-" * 60)
    unique, counts = np.unique(flip_reset_counts, return_counts=True)
    for val, count in zip(unique, counts):
        pct = 100 * count / len(flip_reset_counts)
        print(f"  {val} flip resets: {count} episodes ({pct:.1f}%)")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()


