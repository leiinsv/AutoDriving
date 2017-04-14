from RoomEnv import RoomEnv
from TFBrain import TFBrain
import random
import numpy as np
from tqdm import tqdm

def play():
    roomEnv = RoomEnv()
    brain_config = {
        'num_actions': roomEnv.get_num_actions(),
        'state_dimensions': roomEnv.get_state_dimensions(),
        'experience_size': 300,
        'start_learn_threshold': 50,
        'gamma': 0.7,
        'learning_steps_total': 10000,
        'learning_steps_burin': 100,
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
    for i in tqdm(range(max_steps)):
        action = brain.decide(state)
#        reward, new_state, valid = roomEnv.react(state, action, determistic=True)
        reward, new_state, valid = roomEnv.react(state, action, determistic=False)
        endstate = roomEnv.is_endstate(new_state)
        chain_end = endstate or (not valid)
        brain.learn(state, action, reward, new_state, chain_end)

        if valid:
            state = new_state
            if endstate:
                state = roomEnv.reset_state()
        else:
            state = roomEnv.get_state()

    for i in range(0, roomEnv.get_state_dimensions()):
        test_state = np.zeros(roomEnv.get_state_dimensions())
        test_state[i] = 1
        test_action = brain.decide(test_state, determistic=True)
        print("%d\t%d" % (i, np.argmax(test_action)))


def main():
    play()

if __name__ == '__main__':
    main()  



