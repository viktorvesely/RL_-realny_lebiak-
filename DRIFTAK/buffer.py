import numpy as np

class Buffer():
    def __init__(self, buffer_size, state_space, action_space, batch_size) -> None:

        self.state_dimensions = state_space.shape[0]
        self.action_dimensions = action_space.shape[0]
        self.max_dimension = max(self.state_dimensions, self.action_dimensions)
        self.buffer_size = buffer_size

        self.batch_size = batch_size
        self.first_empty = 0
        self.full = False

        state = np.array(np.zeros(self.state_dimensions))
        reward = np.array(0)
        action = np.array(np.zeros(self.action_dimensions))
        done = np.array(0)
        row = [np.array([state, action, reward, state, done], dtype=object)]
        self.memory = np.repeat(row, self.buffer_size, axis=0)
    
    def record(self, state, action, reward, new_state, done):

        self.memory[self.first_empty, 0] = np.array(state, dtype=float)
        self.memory[self.first_empty, 1] = np.array(action, dtype=float)
        self.memory[self.first_empty, 2] = np.array(reward, dtype=float)
        self.memory[self.first_empty, 3] = np.array(new_state, dtype=float)
        self.memory[self.first_empty, 4] = np.array(done, dtype=float)

        self.first_empty += 1

        if self.first_empty == self.buffer_size:
            self.full = True

        self.first_empty %= self.buffer_size

    def __call__(self) -> tuple:

        number_of_rows = self.buffer_size
        random_indices = np.random.choice(number_of_rows, size = self.batch_size, replace=False)

        batch = self.memory[random_indices]

        state = batch[:, 0]
        action = batch[:, 1]
        reward = batch[:, 2]
        new_state = batch[:, 3]
        done = batch[:, 4]

        return state, action, reward, new_state, done
    
    def __len__(self):
        return self.buffer_size if self.full else self.first_empty