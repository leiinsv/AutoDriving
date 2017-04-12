from TFBrain import TFBrain
import random
import numpy as np

class RoomEnv():
    """
    Using the "Robot in a Room" example in MIT auto-driving class as a simple case to test TFBrain
    Please refer the video starting from 26:52 for the details of the example
    https://www.youtube.com/watch?v=QDzM8r3WgBw
    """
    def __init__(self):
        # a room depicted by a matrix
        self.num_rows = 3
        self.num_cols = 4
        # 0 move up, 1 move left, 2 move right
        self.actions = [0, 1, 2] 
        # default reward per move
        self.step_reward = -0.04

        # initial position is in left bottom unit
        self.state = (self.num_rows-1, 0)

    def _get_2d_axis(self, state):
        idx = np.argmax(state)
        row = int(idx / self.num_cols)
        col = idx % self.num_cols
        return row, col

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

    def _is_valid(self, row, col):
        valid = True
        # out of room range
        if row < 0 or row > self.num_rows - 1:
            valid = False
        if col < 0 or col > self.num_cols -1:
            valid = False
        # internal barrier block
        if row == 1 and col == 1:
            valid = False
        return valid

    def react(self, state, action, determistic=False):
        reward = 0.0
        new_state = np.zeros_like(state)
        # when to move up, in result 10% moves left, 10% right, and 80% moves up
        action_idx = np.argmax(action)
        if (not determistic) and (action_idx == 0):
            prob = random.random()
            if prob <= 0.1:
                action_idx = 1
            elif prob <= 0.2:
                action_idx = 2
            else:
                action_idx = 0
        
        row, col = self._get_2d_axis(state)
        if action_idx == 0:
            row -= 1
        elif action_idx == 1:
            col -= 1
        elif action_idx == 2:
            col += 1
        else:
            pass

        valid = self._is_valid(row, col)
        if valid:
            self.state = (row, col)
            new_state = self.get_state()
            reward += self.step_reward
            if (row == 0) and (col == 3):
                reward += 1
            if (row == 1) and (col == 3):
                reward += -1

        return reward, new_state, valid
        
def test_room():        
    roomEnv = RoomEnv()
    state = np.zeros(12)
    state[4] = 1
    action = np.zeros(3)
    action[2] = 1
    reward, new_state, valid = roomEnv.react(state, action, determistic=True)
    print(reward)
    print(new_state)
    print(valid)

    

def play():
    roomEnv = RoomEnv()
    brain_config = {
        'num_actions': roomEnv.get_num_actions(),
        'state_dimensions': roomEnv.get_state_dimensions(),
        'experience_size': 100,
        'start_learn_threshold': 100,
        'gamma': 0.7,
        'learning_steps_total': 500000,
        'learning_steps_burin': 1000,
        'epsilon_min': 0.0,
        'epsilon_test_time': 0.0,
        'learning_rate': 0.01,
        'batch_size': 50
    }    
    brain = TFBrain(brain_config, learning=True)

    state = roomEnv.get_state()
    max_steps = 500000
    i = 0
    while i < max_steps:
        action = brain.decide(state)
        reward, new_state, valid = roomEnv.react(state, action)
        if valid:
            brain.learn(state, action, reward, new_state)
            state = new_state
        i += 1
    
    for i in range(0, roomEnv.get_state_dimensions()):
        test_state = np.zeros(roomEnv.get_state_dimensions())
        test_state[i] = 1
        test_action = brain.decide(test_state, determistic=True)
        print("%d\t%d" % (i, np.argmax(test_action)))

def main():
    play()
    
if __name__ == '__main__':
    main()  



