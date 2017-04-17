import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from TFBrain import TFBrain
from TFBrain import Experience
from tqdm import tqdm


# Preprocess raw image to 80*80 image

def preprocess(img):
    img = cv2.resize(img, (80,80))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    img = np.reshape(img, (80,80,1))
    return img

def test_tfbrain():
    brain_config = {
        'network_type': 'cnn',
        'learning': True,
        'num_actions': 2,
        'experience_size': 50000,
        'start_learn_threshold': 100,
        'gamma': 0.7,
        'learning_steps_total': 200000,
        'learning_steps_burin': 1000,
        'epsilon_min': 0.0,
        'epsilon_test_time': 0.0,
        'learning_rate': 0.001,
        'batch_size': 32
    }

    brain = TFBrain(brain_config)
    brain.show_configs()
    
    bird_env = game.GameState()
    
    action = np.array([0,1])
    state, reward, chain_end = bird_env.frame_step(action)
    state = preprocess(state)

    while 1!= 0:
        action = brain.decide(state, determistic=False)
        next_state, reward, chain_end = bird_env.frame_step(action)
        next_state = preprocess(next_state)
        experience = Experience(state, action, reward, next_state, chain_end)
        brain.learn(experience)
        state = next_state

def main():
    test_tfbrain()

if __name__ == '__main__':
    main()  



