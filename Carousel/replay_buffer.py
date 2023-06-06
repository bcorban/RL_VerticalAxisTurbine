from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer


class CustomReplayBuffer(ReplayBuffer): #Replay buffer wrapper to ignore transitions during the first few actuated rotations
    def add(self, data):
        # Check the condition based on the state information
        if not self.should_store_transition(data):
            super().add(data)

    def should_store_transition(self, data):
        # Implement your condition here based on the state information
        # For example, if the position is in the state, you can check if it meets your criteria
        return data.__getitem__('default_policy').__getitem__('infos')[0]['transient']
    