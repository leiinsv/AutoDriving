import random
import numpy as np

class RoomEnv():
    """
    Using the "Robot in a Room" example in MIT auto-driving class as a simple case to test TFBrain
    Please refer the video starting from 26:52 for the details of the example
    https://www.youtube.com/watch?v=QDzM8r3WgBw
    """
    def __init__(self, step_reward=-0.04):
        # a room depicted by a matrix
        self.num_rows = 3
        self.num_cols = 4
        # 0 move up, 1 move down, 2 move left, 3 move right
        self.actions = [0, 1, 2, 3] 
        # default reward per move
        self.step_reward = step_reward
        # lasted valid state, initial position is in left bottom unit
        self.state = (self.num_rows-1, 0)
        self.initial_state = (self.num_rows-1, 0)
        self.determistic = True

    def react(self, state, action, determistic=False):
        self.determistic = determistic
        reward = 0.0
        next_state = np.zeros_like(state)
        action_idx = np.argmax(action)

        if (not determistic) and (action_idx == 0):
            prob = random.random()
            if prob <= 0.1:
                action_idx = 2
            elif prob <= 0.2:
                action_idx = 3
            else:
                action_idx = 0
                        
        row, col = self._get_2d_axis(state)
        if action_idx == 0:
            row -= 1
        elif action_idx == 1:
            row += 1
        elif action_idx == 2:
            col -= 1
        else:
            col += 1

        valid = self._is_valid(row, col)
        if valid:
            self.state = (row, col)
            next_state = self.get_state()
            reward = self.step_reward
            if (row == 0) and (col == 3):
                reward = 1
            if (row == 1) and (col == 3):
                reward = -1
            
        return reward, next_state, valid

    def get_num_actions(self):
        return len(self.actions)

    def get_state_dimensions(self):
        return self.num_rows * self.num_cols
    
    # fit the input format of TFBrain
    def get_state(self):
        state = np.zeros(self.num_rows * self.num_cols)
        idx = self.state[0] * self.num_cols + self.state[1]
        state[idx] = 1
        return state

    def get_initial_state(self):
        state = np.zeros(self.num_rows * self.num_cols)
        idx = self.initial_state[0] * self.num_cols + self.initial_state[1]
        state[idx] = 1
        return state
    
    def end_state(self, state):
        row, col = self._get_2d_axis(state)
        if row == 0 and col == 3:
            return True
        elif row == 1 and col == 3:
            return True
        else:
            return False

    def reset_state(self):
        state = self.get_state()
        if self.end_state(state):
           return self.get_initial_state()
        else:
            return state

    def evaluate(self, policy):
        accuracy = 0
        # Set the actions on end/invalid states to be -1
        policy[3] = -1
        policy[5] = -1
        policy[7] = -1
        optimal = []
        if self.determistic and (abs(self.step_reward - (-0.04)) <= 1e-6):
            optimal = [3, 3, 3, -1, 0, -1, 0, -1, 3, 3, 0, 2]
        if (not self.determistic) and (abs(self.step_reward - (-0.04)) <= 1e-6): 
            optimal = [3, 3, 3, -1, 0, -1, 0, -1, 0, 2, 2, 2]
        for (a,b) in zip(optimal, policy):
            if a == b:
                accuracy += 1.0 / len(optimal)
        return accuracy

    def _get_2d_axis(self, state):
        idx = np.argmax(state)
        row = int(idx / self.num_cols)
        col = idx % self.num_cols
        return row, col

    def _is_valid(self, row, col):
        valid = True
        # out of room range
        if row < 0 or row > self.num_rows - 1:
            valid = False
        if col < 0 or col > self.num_cols - 1:
            valid = False
        # internal barrier block
        if row == 1 and col == 1:
            valid = False
        return valid


