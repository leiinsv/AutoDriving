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
        # 0 move up, 1 move down, 2 move left, 3 move right
        self.actions = [0, 1, 2, 3] 
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
        if col < 0 or col > self.num_cols - 1:
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
        intended = True
        if (not determistic) and (action_idx == 0):
            prob = random.random()
            if prob <= 0.1:
                action_idx = 2
                intended = False
            elif prob <= 0.2:
                action_idx = 3
                intended = False
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
            new_state = self.get_state()
            reward += self.step_reward
            if (row == 0) and (col == 3):
                reward = 1
            if (row == 1) and (col == 3):
                reward = -1
        elif intended:
            reward = -100
        else:
            pass
#            reward = self.step_reward
        return reward, new_state, valid

    def is_endstate(self, state):
        row, col = self._get_2d_axis(state)
        if row == 0 and col == self.num_cols - 1:
            return True
        elif row == 1 and col == self.num_cols -1:
            return True
        else:
            return False
    
    def reset_state(self):
#        self.state = [2,0]
#        return self.get_state()
        fit = False
        while not fit:
            row = random.randrange(self.num_rows)
            col = random.randrange(self.num_cols)
            fit = self._is_valid(row, col)
            self.state = (row, col)
            if self.is_endstate(self.get_state()):
                fit = False
        return self.get_state()

        
def transit(s_idx, a_idx):
    roomEnv = RoomEnv()
    state = np.zeros(12)
    state[s_idx] = 1
    action = np.zeros(4)
    action[a_idx] = 1
    reward, new_state, valid = roomEnv.react(state, action, determistic=True)
    ns_idx = np.argmax(new_state)
    return ns_idx, reward, valid
    
def test_room():        
    s_idx = 0
    a = 0
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is False
    a = 1
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is True
    assert ns_idx == 4
    assert abs(reward - (-0.04)) < 1e-6
    a = 2
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is False
    a = 3
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is True
    assert ns_idx == 1
    assert abs(reward - (-0.04)) < 1e-6 

    s_idx = 11
    a = 0
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is True
    assert ns_idx == 7
    assert abs(reward - (-1.0)) < 1e-6
    a = 1
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is False
    a = 2
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is True
    assert ns_idx == 10
    assert abs(reward - (-0.04)) < 1e-6
    a = 3
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is False

    s_idx = 4
    a = 3
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is False

    s_idx = 7
    a = 0
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is True
    assert ns_idx == 3
    assert abs(reward - 1) < 1e-6 
    a = 1
    ns_idx, reward, valid = transit(s_idx, a)
    assert valid is True
    assert ns_idx == 11
    assert abs(reward - (-0.04)) < 1e-6 


def play():
    roomEnv = RoomEnv()
    brain_config = {
        'num_actions': roomEnv.get_num_actions(),
        'state_dimensions': roomEnv.get_state_dimensions(),
        'experience_size': 3000,
        'start_learn_threshold': 500,
        'gamma': 0.7,
        'learning_steps_total': 10000,
        'learning_steps_burin': 1000,
        'epsilon_min': 0.0,
        'epsilon_test_time': 0.0,
        'learning_rate': 0.001,
        'batch_size': 64
    }    
    brain = TFBrain(brain_config, learning=True)

    state = roomEnv.get_state()
    print(state)
    max_steps = 10000
    i = 0
    while i < max_steps:
        action = brain.decide(state)
#        reward, new_state, valid = roomEnv.react(state, action, determistic=True)
        reward, new_state, valid = roomEnv.react(state, action, determistic=False)
        is_end = roomEnv.is_endstate(new_state)
        brain.learn(state, action, reward, new_state, is_end)

        if valid:
            state = new_state
            if is_end:
                state = roomEnv.reset_state()
        else:
            state = roomEnv.get_state()
        i += 1

    for k in range(0, 3):
        for i in range(0, roomEnv.get_state_dimensions()):
            test_state = np.zeros(roomEnv.get_state_dimensions())
            test_state[i] = 1
            test_action = brain.decide(test_state, determistic=True)
            print("%d\t%d" % (i, np.argmax(test_action)))
        print("*" * 50)


def main():
    play()
#    test_room()

if __name__ == '__main__':
    main()  



