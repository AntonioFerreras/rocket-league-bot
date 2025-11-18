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






def build_rlgym_v2_env():
    import random
    from typing import Dict, Any, List
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
        GoalReward,
    )
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )
    from rlgym.rocket_league.rlviser import RLViserRenderer

    from math_utils import dir_to_euler_yzx
    from rewards import (
        DistancePlayerToBallReward,
        BallToGoalReward,
        DistancePlayerToGround,
        VelocityPlayerToBallReward,
        TouchReward,
        ForwardBiasReward,
        ZoneReward,
        BallZoneReward,
        PlayerFallPunishment,
        BoostChangeReward,
    )

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

    goal_reward_weight = 10*0
    touch_reward_weight = 8.0
    distance_player_to_ball_reward_weight = 2.5*0.3
    velocity_player_to_ball_reward_weight = 0.5*0.3
    ball_to_goal_reward_weight = 1.5 * 0
    distance_player_to_ground_reward_weight = 1.5*0
    forward_bias_reward_weight = 0.5*0
    zone_reward_weight = 1.0
    ball_zone_reward_weight = 2.0
    player_fall_punishment_weight = 5.0*0
    boost_change_reward_weight = 0.5*0
    
    reward_fn = CombinedReward(
        (GoalReward(), goal_reward_weight),
        (TouchReward(), touch_reward_weight),
        (DistancePlayerToBallReward(), distance_player_to_ball_reward_weight),
        (VelocityPlayerToBallReward(), velocity_player_to_ball_reward_weight),
        (BallToGoalReward(), ball_to_goal_reward_weight),
        (DistancePlayerToGround(), distance_player_to_ground_reward_weight),
        (ForwardBiasReward(), forward_bias_reward_weight),
        (ZoneReward(), zone_reward_weight),
        (PlayerFallPunishment(), player_fall_punishment_weight),
        (BoostChangeReward(), boost_change_reward_weight),
        (BallZoneReward(), ball_zone_reward_weight),
    )

    obs_builder = DefaultObs(
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

    class AirDribbleMutator(StateMutator[GameState]):
        """
        A StateMutator that sets up the game state for a kickoff.
        """


        def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
            state.config.boost_consumption = 0.001

            spawn_min_x = -3200
            spawn_max_x = 3200

            spawn_min_y = -2000
            spawn_max_y = 3600
            
            spawn_min_z = 700
            spawn_max_z = 1800

            car_min_height_under_ball = 100 # 500
            car_max_height_under_ball = 800 # 1000
            car_x_radius = 50 # 800
            car_y_min = -50 # -800
            car_y_max = 50 # -800
            car_dir_noise_radius = 0*np.pi/16

            ball_vel_xy_noise_radius = 400
            ball_vel_z_min = 50
            ball_vel_z_max = 650

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
            

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        AirDribbleMutator(),
    )
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer(),
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
        PPOMetricsLogger,
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

    # The obs_space_type and action_space_type are determined by your choice of ObsBuilder and ActionParser respectively.
    # The logic used here assumes you are using the types defined by the DefaultObs and LookupTableAction above.
    DefaultObsSpaceType = Tuple[str, int]
    DefaultActionSpaceType = Tuple[str, int]

    train_dtype = torch.bfloat16

    def actor_factory(
        obs_space: DefaultObsSpaceType,
        action_space: DefaultActionSpaceType,
        device: str,
    ):
        dim = 512
        num_layers = 3
        return DiscreteFF(
            obs_space[1], 
            action_space[1], 
            (dim,) * num_layers, 
            device, 
            dtype=train_dtype
        )

    def critic_factory(obs_space: DefaultObsSpaceType, device: str):
        dim = 512
        num_layers = 3
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
            ),
            timestep_limit=60_000_000_000,  # Train for 60B steps
        ),
        process_config=ProcessConfigModel(
            n_proc=64 if not args.render else 1,  # Number of processes to spawn to run environments. Increasing will use more RAM but should increase steps per second, up to a point
            render=args.render,
            render_delay=0 if not args.render else action_repeat/120.0/render_speed,
        ),
        agent_controllers_config={
            "PPO1": PPOAgentControllerConfigModel(
                checkpoint_load_folder=args.resume_ckpt,
                timesteps_per_iteration=370_000,
                learner_config=PPOLearnerConfigModel(
                    batch_size=200_000,
                    ent_coef=0.01,  # Sets the entropy coefficient used in the PPO algorithm
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

    learning_coordinator = LearningCoordinator(
        build_rlgym_v2_env,
        agent_controllers={
            "PPO1": PPOAgentController(
                actor_factory=actor_factory,
                critic_factory=critic_factory,
                experience_buffer=NumpyExperienceBuffer(GAETrajectoryProcessor()),
                metrics_logger=WandbMetricsLogger(PPOMetricsLogger()),
                obs_standardizer=None,
            )
        },
        config=config,
    )
    learning_coordinator.start()