from TFBrain import TFBrain
from random import random
import numpy as np

"""
class MPEnv():
    # Implement logics of reward in the environment of motion planning
    def __init__(self):
        self.current_state = None
        self.num_actions = 5
        self.state_dimensions = 250
        
    def react(self, state, action):
        pass
    
    def get_state(self):
        return self.current_state
    
    def get_num_actions(self):
        return self.num_actions
    
    def get_state_dims(self):
        return self.state_dimensions

"""
class RoomEnv():
    def __init__(self):
        self.num_actions = 3 # 0 up, 1 left, 2 right
        self.state_dimensions = 12
        self.num_room_rows = 3
        self.num_room_cols = 4
        self.step_reward = -0.04
        
        self.initial_state = np.zeros(self.num_room_rows * self.num_room_cols)
        initial_position = (self.num_room_rows - 1) * self.num_room_cols
        self.initial_state[initial_position] = 1

    def _get_2d_axis(self, state):
        idx = np.argmax(state)
        row = int(idx / self.num_cols)
        col = idx % self.num_cols
        return row, col

    def is_valid_state(self, row, col):
        valid = True
        if row < 0 or row > self.num_room_rows - 1:
            valid = False
        if col < 0 or col > self.num_room_cols -1:
            valid = False
        if row == 1 and col == 1:
            valid = False
        return valid

    def get_state(self):
        return self.initial_state

    def react(self, state, action, determistic=False):
        reward = 0.0
        new_state = np.zeros_like(state)

        if (not determistic) and (action == 0):
            prob = random.random()
            if prob <= 0.1:
                action = 1
            elif prob <= 0.2:
                action = 2
            else:
                action = 0
        
        row, col = self._get_2d_axis(state)
        if action == 0:
            row -= 1
        elif action == 1:
            col -= 1
        elif action == 2:
            col += 1
        else:
            pass

        valid = self._valid_state(row, col)
        if valid:
            new_state[row][col] = 1
            reward += self.step_reward
            if (row == 0) and (col == 3):
                reward += 1
            if (row == 1) and (col == 3):
                reward += -1

        return reward, new_state, valid
        
def play():        
    roomEnv = RoomEnv()
    print(roomEnv.get_state())



"""
def play():
    mpEnv = MPEnv()
    brain_config = {
        'num_actions': mpEnv.get_num_actions(),
        'state_dimensions': mpEnv.get_state_dims(),
        'experience_size': 3000,
        'start_learn_threshold': 500,
        'gamma': 0.7,
        'learning_steps_total': 10000,
        'learning_steps_burnin': 1000,
        'epsilon_min': 0.0,
        'epsilon_test_time': 0.0,
        'learning_rate': 0.001,
        'momentum': 0.0,
        'batch_size': 64
    }    
    brain = TFBrain(brain_config, learning=True)
    state = mpEnv.get_state()
    max_steps = 50000
    i = 0
    while i < max_steps:
        action = brain.decide(state)
        reward, new_state, valid = mpEnv.react(state, action)
        if valid:
            brain.learn(state, action, reward, new_state)
            state = new_state
        i += 1
"""

def main():
    play()
    
if __name__ == '__main__':
    main()  


# In[ ]:



