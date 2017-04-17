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

def init_state(observation):
    state = np.stack((observation, observation, observation, observation), axis = 2)
    return state

def proceed_state(state, next_observation):
    next_state = np.append(state[:,:,1:], next_observation, axis = 2)
    return next_state

def test_tfbrain():
    brain_config = {
        'network_type': 'cnn',
        'learning': True,
        'num_actions': 2,
        'experience_size': 50000,
        'gamma': 0.99,
        'observe_age': 100,
        'explore_age': 350000,
        'explore_burin': 200,
        'epsilon_min': 0.0,
        'epsilon_test_time': 0.0,
        'batch_size': 32,
        'lookback_window': 3
    }

    brain = TFBrain(brain_config)
    brain.show_configs()
    
    bird_env = game.GameState()
    frame_per_action = 1
    
    action = np.array([1,0])
    observation, reward, chain_end = bird_env.frame_step(action)
    observation = preprocess(observation)
    observation = np.reshape(observation, (observation.shape[0], observation.shape[1]))
    state = init_state(observation)

    i = 0
    while 1!= 0:
        if i % frame_per_action == 0:
            action = brain.decide(state, determistic=False)
        else:
            # Do nothing
            action = np.array([1,0])
        next_observation, reward, chain_end = bird_env.frame_step(action)
        next_observation = preprocess(next_observation)
        next_state = proceed_state(state, next_observation)
        experience = Experience(state, action, reward, next_state, chain_end)
        brain.learn(experience)
        state = next_state
        i += 1

def main():
    test_tfbrain()

if __name__ == '__main__':
    main()  



