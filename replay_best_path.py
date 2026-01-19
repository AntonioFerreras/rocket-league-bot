"""
Replay script to visualize the best path found during evaluation.
Loads path from saved .npz file and runs it with rendering.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

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
render_speed = 1.0


def load_path_data(path_file):
    """Load path data from .npz file."""
    data = np.load(path_file, allow_pickle=True)
    
    path_info = {
        "has_setup": bool(data["has_setup"][0]) if len(data["has_setup"]) > 0 else False,
        "ball_spawn": data["ball_spawn"].copy() if len(data["ball_spawn"]) > 0 else None,
        "car_spawn": data["car_spawn"].copy() if len(data["car_spawn"]) > 0 else None,
        "num_setup_points": int(data["num_setup_points"][0]) if len(data["num_setup_points"]) > 0 else 0,
    }
    
    # Load initial physics state if available (for exact replay)
    initial_state = None
    if "ball_position" in data:
        initial_state = {
            "ball_position": data["ball_position"].copy(),
            "ball_linear_velocity": data["ball_linear_velocity"].copy(),
            "ball_angular_velocity": data["ball_angular_velocity"].copy(),
            "car_position": data["car_position"].copy(),
            "car_linear_velocity": data["car_linear_velocity"].copy(),
            "car_angular_velocity": data["car_angular_velocity"].copy(),
            "car_euler_angles": data["car_euler_angles"].copy(),
            "car_boost_amount": float(data["car_boost_amount"][0]),
            "car_on_ground": bool(data["car_on_ground"][0]),
            "car_has_jumped": bool(data["car_has_jumped"][0]),
            "car_air_time_since_jump": float(data["car_air_time_since_jump"][0]) if "car_air_time_since_jump" in data else (0.0 if not bool(data["car_has_jumped"][0]) else 2.0),
        }
    
    # Load expected flip reset count if available
    expected_flip_resets = None
    if "expected_flip_resets" in data:
        expected_flip_resets = int(data["expected_flip_resets"][0])
    
    return {
        "path_points": data["path_points"].copy(),
        "path_start": data["path_start"].copy(),
        "path_end": data["path_end"].copy(),
        "path_control": data["path_control"].copy(),
        "condition_data": data["condition_data"].copy(),
        "path_info": path_info,
        "initial_state": initial_state,
        "expected_flip_resets": expected_flip_resets,
    }


def build_replay_env(path_data):
    """Build environment that uses the pre-loaded path data."""
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
    from rlgym.rocket_league.rlviser import RLViserRenderer
    from path_generator_viz import PathVisualizer
    from math_utils import dir_to_euler_yzx, normalize

    class CompositeRenderer:
        def __init__(self):
            self.rlviser = RLViserRenderer()
            self.path_viz = PathVisualizer(blocking=False)
            self.last_path_points = None
        
        def render(self, state, shared_info):
            self.rlviser.render(state, shared_info)
            
            if "path_points" in shared_info:
                path_points = shared_info["path_points"]
                should_update = False
                if self.last_path_points is None:
                    should_update = True
                elif len(path_points) != len(self.last_path_points):
                    should_update = True
                elif not np.allclose(path_points, self.last_path_points):
                    should_update = True
                
                if should_update:
                    self.last_path_points = path_points.copy()
                    start = shared_info.get("path_start", np.zeros(3))
                    end = shared_info.get("path_end", np.zeros(3))
                    control = shared_info.get("path_control", np.zeros(3))
                    condition_data = shared_info.get("condition_data", None)
                    self.path_viz.show_path(path_points, start, end, control, condition_data)

            if "current_target_index" in shared_info:
                self.path_viz.update_controls(shared_info["current_target_index"])
            
            self.path_viz.update_ball(state.ball.position)
            self.path_viz.process_events()
        
        def close(self):
            pass

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

    # Mutator that uses pre-loaded path data
    class ReplayPathMutator(StateMutator[GameState]):
        def __init__(self, path_data, num_conditions: int):
            self.path_data = path_data
            self.num_conditions = num_conditions

        def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
            state.config.boost_consumption = 0.001
            
            # Use pre-loaded path data
            path_points = self.path_data["path_points"]
            start_point = self.path_data["path_start"]
            end_point = self.path_data["path_end"]
            control_point = self.path_data["path_control"]
            condition_data = self.path_data["condition_data"]
            path_info = self.path_data["path_info"]
            initial_state = self.path_data.get("initial_state")
            
            # If we have saved initial state, use it for exact replay
            if initial_state is not None:
                # Restore ball state exactly
                state.ball.position = initial_state["ball_position"].astype(np.float32)
                state.ball.linear_velocity = initial_state["ball_linear_velocity"].astype(np.float32)
                state.ball.angular_velocity = initial_state["ball_angular_velocity"].astype(np.float32)
                
                # Restore car state exactly
                for car in state.cars.values():
                    car.physics.position = initial_state["car_position"].astype(np.float32)
                    car.physics.linear_velocity = initial_state["car_linear_velocity"].astype(np.float32)
                    car.physics.angular_velocity = initial_state["car_angular_velocity"].astype(np.float32)
                    car.physics.euler_angles = initial_state["car_euler_angles"].astype(np.float32)
                    car.boost_amount = initial_state["car_boost_amount"]
                    car.on_ground = initial_state["car_on_ground"]
                    car.has_jumped = initial_state["car_has_jumped"]
                    car.air_time_since_jump = initial_state["car_air_time_since_jump"]
            else:
                # Fallback: use path_info to set up state (less accurate)
                has_setup = path_info["has_setup"]
                
                if has_setup:
                    ball_spawn = path_info["ball_spawn"]
                    car_spawn = path_info["car_spawn"]
                    
                    state.ball.position = ball_spawn.astype(np.float32)
                    state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
                    state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
                    
                    for car in state.cars.values():
                        car.physics.position = car_spawn.astype(np.float32)
                        to_ball = state.ball.position - car.physics.position
                        to_ball[2] = 0.0
                        car.physics.euler_angles = dir_to_euler_yzx(to_ball)
                        car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
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
                    ball_vel = np.array([0, 0, 350], dtype=np.float32)
                    state.ball.linear_velocity = ball_vel + objective_direction * 200
                    state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
                    
                    ball_x, ball_y, ball_z = state.ball.position
                    for car in state.cars.values():
                        car.physics.position = np.array([ball_x, ball_y + 20, ball_z - 300], dtype=np.float32)
                        to_ball = state.ball.position - car.physics.position
                        to_ball = to_ball / np.linalg.norm(to_ball)
                        car.physics.linear_velocity = to_ball * 350
                        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
                        car.physics.euler_angles = dir_to_euler_yzx(state.ball.position - car.physics.position)
                        car.boost_amount = 100.0
                        car.air_time_since_jump = 2.0
                        car.has_jumped = True
            
            # Store path info in shared_info
            shared_info["path_points"] = path_points.astype(np.float32)
            shared_info["path_start"] = start_point.astype(np.float32)
            shared_info["path_end"] = end_point.astype(np.float32)
            shared_info["path_control"] = control_point.astype(np.float32)
            shared_info["path_info"] = path_info
            shared_info["air_roll_rate"] = 0.0
            shared_info["air_roll_action"] = 0
            shared_info["condition_data"] = condition_data.astype(np.float32)
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
            self.flip_reset_reward = FlipResetReward(debug=True, target_hit_timeout=10.0)  # Debug mode for replay
            self.ball_to_target_reward = BallToTargetReward(print_hits=True)
        
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
        ReplayPathMutator(path_data, num_conditions=num_conditions),
    )
    
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=EvalReward(),
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=CompositeRenderer(),
    )


def main():
    parser = argparse.ArgumentParser(description="Replay best path with rendering")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint folder containing ppo_learner/actor.pt")
    parser.add_argument("--path_file", type=str, default="best_path.npz",
                        help="Path to saved best path .npz file")
    parser.add_argument("--num_replays", type=int, default=-1,
                        help="Number of times to replay (-1 for infinite)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy (default: True)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load path data
    print(f"Loading path from: {args.path_file}")
    path_data = load_path_data(args.path_file)
    print(f"  Path has {len(path_data['path_points'])} points")
    print(f"  Has setup: {path_data['path_info']['has_setup']}")
    print(f"  Has exact initial state: {path_data['initial_state'] is not None}")
    expected_flip_resets = path_data.get("expected_flip_resets")
    if expected_flip_resets is not None:
        print(f"  Expected flip resets: {expected_flip_resets}")
    
    # Build environment
    env = build_replay_env(path_data)
    
    # Load actor model
    from models import DiscreteFF
    
    obs_space = env.observation_space("blue-0")
    action_space = env.action_space("blue-0")
    
    print(f"Obs space: {obs_space}")
    print(f"Action space: {action_space}")
    
    dim = 512
    num_layers = 4
    actor = DiscreteFF(
        obs_space[1],
        action_space[1],
        (dim,) * num_layers,
        device,
        dtype=torch.float32
    )
    
    actor_path = os.path.join(args.checkpoint, "ppo_learner", "actor.pt")
    print(f"Loading actor from: {actor_path}")
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    render_delay = action_repeat / 120.0 / args.speed
    
    replay_count = 0
    print(f"\nStarting replay (speed: {args.speed}x, deterministic: {args.deterministic})...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while args.num_replays < 0 or replay_count < args.num_replays:
            replay_count += 1
            print(f"=== Replay #{replay_count} ===")
            
            obs_dict = env.reset()
            done = False
            step_count = 0
            
            while not done:
                agent_ids = list(obs_dict.keys())
                obs_list = [obs_dict[agent_id] for agent_id in agent_ids]
                
                with torch.no_grad():
                    actions, _ = actor.get_action(agent_ids, obs_list, deterministic=args.deterministic)
                
                action_dict = {agent_id: np.array([actions[i]]) for i, agent_id in enumerate(agent_ids)}
                
                obs_dict, rewards, terminated, truncated = env.step(action_dict)
                
                # Render
                env.render()
                time.sleep(render_delay)
                
                done = any(terminated.values()) or any(truncated.values())
                step_count += 1
            
            # Keep rendering for a moment after episode ends (e.g., after goal)
            print("  Episode ended, showing result...")
            for _ in range(int(120 / action_repeat)):  # ~1 second at 120 ticks/sec
                env.render()
                time.sleep(render_delay)
            
            # Print episode stats
            shared_info = env.shared_info
            actual_flip_resets = shared_info.get('num_flip_resets', 0)
            print(f"  Steps: {step_count}")
            print(f"  Flip resets: {actual_flip_resets}", end="")
            if expected_flip_resets is not None:
                if actual_flip_resets == expected_flip_resets:
                    print(f" (matches expected: {expected_flip_resets}) ✓")
                else:
                    print(f" (MISMATCH! expected: {expected_flip_resets}) ✗")
            else:
                print()
            print(f"  Ball touches: {shared_info.get('num_ball_touches', 0)}")
            print()
            
            # Brief pause between replays
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nStopping replay...")
    
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()

