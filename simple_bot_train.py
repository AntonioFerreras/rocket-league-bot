import os


# needed to prevent numpy from using a ton of memory in env processes and causing them to throttle each other
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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






def build_rlgym_v2_env(debug=False):
    import random
    from typing import Dict, Any, List, Tuple
    import numpy as np
    from rlgym.rocket_league.common_values import BALL_RESTING_HEIGHT, BLUE_TEAM
    from rlgym.api import RLGym, StateMutator, DoneCondition, AgentID
    from rlgym.rocket_league.api import GameState
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        AnyCondition,
        GoalCondition,
        TimeoutCondition,
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import (
        CombinedReward,
    )
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )
    from rlgym.rocket_league.rlviser import RLViserRenderer
    from path_generator_viz import PathVisualizer

    class CompositeRenderer:
        def __init__(self):
            self.rlviser = RLViserRenderer()
            self.path_viz = PathVisualizer(blocking=False)
            self.last_path_points = None
        
        def render(self, state, shared_info):
            self.rlviser.render(state, shared_info)
            
            # Check if path has changed or needs update
            if "path_points" in shared_info:
                 path_points = shared_info["path_points"]
                 # Simple check: if length or first point changed
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
                     self.path_viz.show_path(path_points, start, end, control)
            
            self.path_viz.update_ball(state.ball.position)
            self.path_viz.process_events()
            # Also handle window close? 
            # self.path_viz.root.protocol("WM_DELETE_WINDOW", ...)

    from math_utils import dir_to_euler_yzx
    from rewards import (
        GoalReward,
        DistancePlayerToBallReward,
        BallToGoalReward,
        DistancePlayerToGround,
        VelocityPlayerToBallReward,
        TouchReward,
        ForwardBiasReward,
        ZoneReward,
        BallZoneReward,
        BoostChangeReward,
        BallToTargetReward,
    )
    from mutators import AirDribbleMutator, AirDribbleDirectedMutator

    

    class NoTouchTimeoutCondition(DoneCondition[AgentID, GameState]):
        """
        A DoneCondition that is satisfied when no car has touched the ball for a specified amount of time.
        Timer starts when the ball is touched for the first time.
        """

        def __init__(self, timeout_seconds: float, freeze_start_tick: bool = False):
            """
            :param timeout_seconds: Timeout in seconds
            """
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
        """
        A DoneCondition that is satisfied a few seconds after the ball hits the ground.
        """

        def __init__(self, timeout_seconds: float):
            """
            :param timeout_seconds: Timeout in seconds
            """
            self.timeout_seconds = timeout_seconds
            self.last_hit_ground_tick = None

        def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
            self.last_hit_ground_tick = None

        def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
            if state.ball.position[2] < BALL_RESTING_HEIGHT*1.5:
                self.last_hit_ground_tick = state.tick_count
                done = False
            else:
                if self.last_hit_ground_tick is None:
                    return {agent: False for agent in agents}
                time_elapsed = (state.tick_count - self.last_hit_ground_tick) / common_values.TICKS_PER_SECOND
                done = time_elapsed >= self.timeout_seconds

            return {agent: done for agent in agents}

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds, freeze_start_tick=False),
        TimeoutCondition(timeout_seconds=game_timeout_seconds), 
        BallHitGroundTimeoutCondition(timeout_seconds=ball_hit_ground_timeout_seconds),
    )

    goal_reward_weight = 6.0
    touch_reward_weight = 2.0
    distance_player_to_ball_reward_weight = 1.5 *0.3
    velocity_player_to_ball_reward_weight = 0.5 *0.3
    ball_to_goal_reward_weight = 1.5 * 0
    distance_player_to_ground_reward_weight = 1.5*0
    forward_bias_reward_weight = 0.5*0
    zone_reward_weight = 1.0
    ball_zone_reward_weight = 2.0
    boost_change_reward_weight = 0.5*0
    ball_to_target_reward_weight = 10.0

    reward_fn = CombinedReward(
        (GoalReward(), goal_reward_weight),
        (TouchReward(), touch_reward_weight),
        (DistancePlayerToBallReward(), distance_player_to_ball_reward_weight),
        (VelocityPlayerToBallReward(), velocity_player_to_ball_reward_weight),
        (BallToGoalReward(), ball_to_goal_reward_weight),
        (DistancePlayerToGround(), distance_player_to_ground_reward_weight),
        (ForwardBiasReward(), forward_bias_reward_weight),
        (ZoneReward(), zone_reward_weight),
        (BoostChangeReward(), boost_change_reward_weight),
        (BallZoneReward(), ball_zone_reward_weight),
        (BallToTargetReward(print_hits=debug), ball_to_target_reward_weight),
    )

    class FreestyleObs(DefaultObs):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_conditions = 16

        def get_obs_space(self, agent: AgentID) -> Tuple[str, int]:
            if self.zero_padding is not None:
                return 'real', 52 + 20 * self.zero_padding * 2 + self.num_conditions
            elif self._state is not None:
                return 'real', 52 + 20 * len(self._state.cars) + self.num_conditions
            else:
                return 'real', -1 # Without zero padding this depends on the current state, but we don't have one yet

        def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
            self._state = state

            obs = {}
            for agent in agents:
                obs[agent] = self._build_obs(agent, state, shared_info)


            # track some stats
            # height of ball
            shared_info["ball_x"] = state.ball.position[0]
            shared_info["ball_y"] = state.ball.position[1]
            shared_info["ball_z"] = state.ball.position[2]

            
            return obs

        def _build_obs(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> np.ndarray:
            obs = super()._build_obs(agent, state, shared_info)
            # add conditions 
            target_pos = shared_info["path_points"][shared_info["current_target_index"]]
            num_targets = len(shared_info["path_points"])
            if shared_info["current_target_index"] + 1 < num_targets:
                next_target_pos = shared_info["path_points"][shared_info["current_target_index"] + 1]
            else:
                next_target_pos = target_pos

            obs = np.concatenate([obs, target_pos*self.POS_COEF])
            obs = np.concatenate([obs, next_target_pos*self.POS_COEF])
            obs = np.concatenate([obs, np.zeros(self.num_conditions-6)])
            return obs

    obs_builder = FreestyleObs(
        zero_padding=pad_team_size,
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    random.seed(123123)
    
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        AirDribbleDirectedMutator(),
    )
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=CompositeRenderer(),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--run_name", type=str, default="airdribble-bot")
    args = parser.parse_args()
    
    from typing import Tuple

    import torch
    import numpy as np
    from rlgym_learn_algos.logging import (
        WandbMetricsLogger,
        WandbMetricsLoggerConfigModel,
    )
    from rlgym_learn_algos.ppo import (
        ExperienceBufferConfigModel,
        GAETrajectoryProcessor,
        GAETrajectoryProcessorConfigModel,
        NumpyExperienceBuffer,
        PPOAgentController,
        PPOAgentControllerConfigModel,
        PPOLearnerConfigModel,
        # PPOMetricsLogger,
    )

    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        NumpySerdeConfig,
        ProcessConfigModel,
        PyAnySerdeType,
        SerdeTypesModel,
        generate_config,
    )
    from rlgym_learn.rocket_league import GameStatePythonSerde

    from models import BasicCritic, DiscreteFF
    from metrics_logger import PPOMetricsLogger

    # The obs_space_type and action_space_type are determined by your choice of ObsBuilder and ActionParser respectively.
    # The logic used here assumes you are using the types defined by the DefaultObs and LookupTableAction above.
    DefaultObsSpaceType = Tuple[str, int]
    DefaultActionSpaceType = Tuple[str, int]

    train_dtype = torch.float32

    def actor_factory(
        obs_space: DefaultObsSpaceType,
        action_space: DefaultActionSpaceType,
        device: str,
    ):
        dim = 512
        num_layers = 4
        return DiscreteFF(
            obs_space[1], 
            action_space[1], 
            (dim,) * num_layers, 
            device, 
            dtype=train_dtype
        )

    def critic_factory(obs_space: DefaultObsSpaceType, device: str):
        dim = 512
        num_layers = 4
        return BasicCritic(
            obs_space[1], 
            (dim,) * num_layers, 
            device, 
            dtype=train_dtype
        )

    # Create the config that will be used for the run
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
                }),
            ),
            timestep_limit=60_000_000_000,  # Train for 60B steps
        ),
        process_config=ProcessConfigModel(
            n_proc=128 if not args.render else 1,  # Number of processes to spawn to run environments. Increasing will use more RAM but should increase steps per second, up to a point
            render=args.render,
            render_delay=0 if not args.render else action_repeat/120.0/render_speed,
        ),
        agent_controllers_config={
            "PPO1": PPOAgentControllerConfigModel(
                checkpoint_load_folder=args.resume_ckpt,
                add_unix_timestamp=False,
                save_every_ts=10_000_000,
                n_checkpoints_to_keep=10,
                timesteps_per_iteration=370_000,
                learner_config=PPOLearnerConfigModel(
                    batch_size=200_000,
                    ent_coef=0.004,  # Sets the entropy coefficient used in the PPO algorithm
                    actor_lr=4e-4,  # Sets the learning rate of the actor model
                    critic_lr=4e-4,  # Sets the learning rate of the critic model
                ),
                experience_buffer_config=ExperienceBufferConfigModel(
                    max_size=1_000_000,  # Sets the number of timesteps to store in the experience buffer. Old timesteps will be pruned to only store the most recently obtained timesteps.
                    trajectory_processor_config=GAETrajectoryProcessorConfigModel(),
                ),
                metrics_logger_config=WandbMetricsLoggerConfigModel(
                    group="rlgym-learn-testing",
                    run=args.run_name
                ),
            )
        },
        agent_controllers_save_folder="agent_controllers_checkpoints",  # (default value) WARNING: THIS PROCESS MAY DELETE ANYTHING INSIDE THIS FOLDER. This determines the parent folder for the runs for each agent controller. The runs folder for the agent controller will be this folder and then the agent controller config key as a subfolder.
    )

    # Generate the config file for reference (this file location can be
    # passed to the learning coordinator via config_location instead of defining
    # the config object in code and passing that)
    generate_config(
        learning_coordinator_config=config,
        config_location="config.json",
        force_overwrite=True,
    )

    from functools import partial
    
    learning_coordinator = LearningCoordinator(
        partial(build_rlgym_v2_env, debug=args.render),
        agent_controllers={
            "PPO1": PPOAgentController(
                actor_factory=actor_factory,
                critic_factory=critic_factory,
                experience_buffer=NumpyExperienceBuffer(GAETrajectoryProcessor()),
                metrics_logger=WandbMetricsLogger(PPOMetricsLogger()) if not args.render else None,
                obs_standardizer=None,
            )
        },
        config=config,
    )
    learning_coordinator.start()