from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
import stable_baselines3.common.buffers
import numpy as np
class CustomReplayBuffer(
    ReplayBuffer
):  # Replay buffer wrapper to ignore transitions during the first few actuated rotations
    def add(self, data):
        # Check the condition based on the state information
        if not self.should_store_transition(data):
            super().add(data)

    def should_store_transition(self, data):
        # Implement condition based on the info dict of a step transition
        return data.__getitem__("default_policy").__getitem__("infos")[0]["transient"]

class SB3_CustomReplayBuffer(stable_baselines3.common.buffers.ReplayBuffer):
    def add(self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos,
    ) -> None:
        # Check the condition based on the state information

        if self.should_store_transition(infos):
            super().add(obs,next_obs,action,reward,done,infos)

    def should_store_transition(self, infos):
        # print(infos)
        # Implement condition based on the info dict of a step transition
        return not infos[0]['transient']