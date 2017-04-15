import random
import numpy as np

class RoomEnv():
    """Using the "Robot in a Room" example in MIT auto-driving class as a simple case to test TFBrain
    Please refer the video starting from 26:52 for the details of the example
    https://www.youtube.com/watch?v=QDzM8r3WgBw
   
    """
    def __init__(self, step_reward=-0.04):
        # A room depicted by a matrix
        self.num_rows = 3
        self.num_cols = 4
        # 0 move up, 1 move down, 2 move left, 3 move right
        self.actions = [0, 1, 2, 3] 
        # Default reward per move
        self.step_reward = step_reward
        # Lasted valid state, initial position is in left bottom unit
        self.state = (self.num_rows-1, 0)
        self.initial_state = (self.num_rows-1, 0)
        # Whether an action will result in determistic state
        self.determistic = True

    def react(self, state, action, determistic=False):
        """Given a state and action, return the next_state and corresponding reward
        @param state
        @param action
        @param determistic - whether the action result in determistic next_state
        @return reward - reward corresponding to next_state
        @return next_state
        @return valid - whether the next_state is valid or not. ex. out of the boundary or in barrier blocks of the room
        """
        self.determistic = determistic
        reward = 0.0
        next_state = np.zeros_like(state)
        action_idx = np.argmax(action)

        if (not determistic) and (action_idx == 0):
            # If it is 'move up', 10% times it moves left, 10% times moves right, and 80% times moves up 
            prob = random.random()
            if prob <= 0.1:
                action_idx = 2
            elif prob <= 0.2:
                action_idx = 3
            else:
                action_idx = 0

        # Make the move                
        row, col = self._get_axis(state)
        if action_idx == 0:
            row -= 1
        elif action_idx == 1:
            row += 1
        elif action_idx == 2:
            col -= 1
        else:
            col += 1
        
        # Set the reward
        valid = self.valid(row, col)
        if valid:
            self.state = (row, col)
            next_state = self.get_state()
            reward = self.step_reward
            # 1st end state with reward 1
            if (row == 0) and (col == 3):
                reward = 1
            # 2nd end state with reward -1
            if (row == 1) and (col == 3):
                reward = -1
            
        return reward, next_state, valid

    def get_num_actions(self):
        return len(self.actions)

    def get_state_dimensions(self):
        return self.num_rows * self.num_cols
    
    def get_state(self):
        # Get the current state
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
        # There are two end states -- top right blcok and the block under it
        row, col = self._get_axis(state)
        if (row == 0 and col == 3) or (row == 1 and col == 3):
            return True
        else:
            return False

    def valid(self, row, col):
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

    def reset_state(self):
        state = self.get_state()
        # If it is end state, reset to initial state
        # Else, return to the previous valid state
        if self.end_state(state):
           return self.get_initial_state()
        else:
            return state

    def evaluate(self, policy):
        # Evaluate the edit distance between the policy and optimal policy
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

    def _get_axis(self, state):
        idx = np.argmax(state)
        row = int(idx / self.num_cols)
        col = idx % self.num_cols
        return row, col



