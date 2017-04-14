from RoomEnv import RoomEnv
from TFBrain import TFBrain
from TFBrain import Experience
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
    max_steps = 10000
    for i in tqdm(range(max_steps)):
        action = brain.decide(state, determistic=False)
        reward, next_state, valid = roomEnv.react(state, action, determistic=True)
        chain_end = roomEnv.end_state(next_state) or not valid
        experience = Experience(state, action, reward, next_state, chain_end)

        brain.learn(experience)

        if chain_end:
            state = roomEnv.reset_state()
        else:
            state = next_state

    for i in range(0, roomEnv.get_state_dimensions()):
        test_state = np.zeros(roomEnv.get_state_dimensions())
        test_state[i] = 1
        test_action = brain.decide(test_state, determistic=True)
        print("%d\t%d" % (i, np.argmax(test_action)))


def main():
    play()

if __name__ == '__main__':
    main()  



