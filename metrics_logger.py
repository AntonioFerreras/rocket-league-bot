from typing import Any, Dict, List
import numpy as np
from rlgym_learn_algos.logging import DictMetricsLogger
from rlgym_learn_algos.ppo import PPOAgentControllerData

from rlgym_learn_algos.ppo import GAETrajectoryProcessorData


class PPOMetricsLogger(
    DictMetricsLogger[
        None,
        None,
        PPOAgentControllerData[GAETrajectoryProcessorData],
    ],
):
    def __init__(self):
        self.state_metrics: Dict[str, Any] = {}
        self.agent_metrics: Dict[str, Any] = {}

    def get_metrics(self) -> Dict[str, Any]:
        return {**self.agent_metrics, **self.state_metrics}

    def collect_env_metrics(self, data: List[Dict[str, Any]]):
        """
        Override this function to set self.state_metrics to something else using the data provided.
        The metrics should be nested dictionaries
        """
        target_dists = []
        for shared_info in data:
            if shared_info is None:
                continue
            
            # Get ball pos
            ball_pos = np.array([shared_info.get("ball_x", 0.0), shared_info.get("ball_y", 0.0), shared_info.get("ball_z", 0.0)])
            
            # Get current target pos from path points if available
            path_points = shared_info.get("path_points")
            current_idx = shared_info.get("current_target_index", 0)
            
            if path_points is not None and len(path_points) > current_idx:
                target_pos = path_points[current_idx]
                dist = np.linalg.norm(ball_pos - target_pos)
                target_dists.append(dist)
            # Fallback: if we have target_x/y/z explicitly set

        avg_target_dist = np.mean(target_dists) if target_dists else 0.0
        avg_air_roll_rate = np.mean([shared_info.get("air_roll_rate", 0.0) if shared_info is not None else 0.0 for shared_info in data])

        avg_num_targets_hit = np.mean([shared_info.get("current_target_index", 0) if shared_info is not None else 0.0 for shared_info in data])
        max_num_targets_hit = np.max([shared_info.get("current_target_index", 0) if shared_info is not None else 0.0 for shared_info in data])

        self.state_metrics = {
            "Tracked metrics": {
                "Average ball height": np.mean([shared_info.get("ball_z", 0.0) if shared_info is not None else 0.0 for shared_info in data]),
                "Average distance to target": avg_target_dist,
                "Average number of targets hit": avg_num_targets_hit,
                "Max number of targets hit": max_num_targets_hit,
                "Average air roll rate": avg_air_roll_rate,
            }
        }



    def collect_agent_metrics(
        self, data: PPOAgentControllerData[GAETrajectoryProcessorData]
    ):
        self.agent_metrics = {
            "Timing": {
                "PPO Batch Consumption Time": data.ppo_data.batch_consumption_time,
                "Total Iteration Time": data.iteration_time,
                "Timestep Collection Time": data.timestep_collection_time,
                "Timestep Consumption Time": data.iteration_time
                - data.timestep_collection_time,
                "Collected Steps per Second": data.timesteps_collected
                / data.timestep_collection_time,
                "Overall Steps per Second": data.timesteps_collected
                / data.iteration_time,
            },
            "Timestep Collection": {
                "Cumulative Timesteps": data.cumulative_timesteps,
                "Timesteps Collected": data.timesteps_collected,
            },
            "PPO Metrics": {
                "Average Reward": data.trajectory_processor_data.average_reward,
                "Average Undiscounted Episodic Return": data.trajectory_processor_data.average_undiscounted_episodic_return,
                "Average Return": data.trajectory_processor_data.average_return,
                "Return Standard Deviation": data.trajectory_processor_data.return_standard_deviation,
                "Cumulative Model Updates": data.ppo_data.cumulative_model_updates,
                "Actor Entropy": data.ppo_data.actor_entropy,
                "Mean KL Divergence": data.ppo_data.kl_divergence,
                "Critic Loss": data.ppo_data.critic_loss,
                "SB3 Clip Fraction": data.ppo_data.sb3_clip_fraction,
                "Actor Update Magnitude": data.ppo_data.actor_update_magnitude,
                "Critic Update Magnitude": data.ppo_data.critic_update_magnitude,
            },
        }

    def validate_config(self, config_obj) -> None:
        return None
