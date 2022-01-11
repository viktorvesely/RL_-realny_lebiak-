import numpy as np


class Buffer:
    EXPERIENCE_SIZE = 4

    def __init__(self, n_experiences, batch_size, state_shape, num_actions):

        self.head = 0
        self.n_experiences = n_experiences
        self.experiences = {
            "state": np.zeros((n_experiences, state_shape[0], state_shape[1], state_shape[2]), dtype="float32"),
            "action": np.zeros((n_experiences, num_actions), dtype="float32"),
            "reward": np.zeros(n_experiences, dtype="float32"),
            "next_state": np.zeros((n_experiences, state_shape[0], state_shape[1], state_shape[2]), dtype="float32")
        }
        self.batch_size = batch_size
        self.full = False

    def __len__(self):
        return self.n_experiences if self.full else self.head
        
    
    def record(self, state, action, reward, nextState):

        self.experiences["state"][self.head] = np.array(state)
        self.experiences["action"][self.head] = np.array(action)
        self.experiences["reward"][self.head] = reward
        self.experiences["next_state"][self.head] = np.array(nextState)

        self.head += 1

        if self.head == self.n_experiences:
            self.full = True

        self.head = self.head % self.n_experiences

    def __call__(self, batch_size=None):
        
        n = self.batch_size

        if batch_size is not None:
            n = batch_size

        bufferLen = len(self)
        n = min(bufferLen, n)
        
        indicies = np.random.choice(bufferLen, n, replace=False)

        return (
            self.experiences["state"][indicies],
            self.experiences["action"][indicies],
            self.experiences["reward"][indicies],
            self.experiences["next_state"][indicies]
        )


        