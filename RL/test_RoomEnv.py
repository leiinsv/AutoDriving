from RoomEnv import RoomEnv
import random
import numpy as np

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

def main():
    test_room()
    print("Tests complete!")

if __name__ == '__main__':
    main()  



