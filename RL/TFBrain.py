import random
import numpy as np
import tensorflow as tf

class Experience(object):
    # Transition of (state, action, reward, next_state), with indication of whether ss is end-state  
    def __init__(self, state, action, reward, new_state, chain_end=False):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.chain_end = chain_end

class TFBrain(object):
    """Q-learning network reinforcement learning based on TensorFlow  """  
    
    def __init__(self, config, learning=True):    
        self._init_configs(config)
        self._build_network()
        self.experiences = []  # Replay memory of experiences  
        self.age = 0  # Effective learning steps (backward pass) occurred so far  
        self.learning = learning # Learn model from the scratch or load existing model. 

#    def learn(self, state, action, reward, new_state, chain_end=False):
    def learn(self, experience):
        """ Q-learning implemented by TensorFlow.
        @param experience - Experience

        """
#        e = Experience(state, action, reward, new_state, chain_end)

        if len(self.experiences) < self.experience_size:
            self.experiences.append(experience)
        else:
            replace_index = random.randrange(self.experience_size)
            self.experiences[replace_index] = experience

        if(len(self.experiences) > max(self.start_learn_threshold, self.batch_size)):
            batch = random.sample(self.experiences, self.batch_size)

            state_batch = [e.state for e in batch]
            action_batch = [e.action for e in batch]
            reward_batch = [e.reward for e in batch]
            new_state_batch = [e.new_state for e in batch]
            chain_end_batch = [e.chain_end for e in batch]

            q_value_fact_batch = []

            # forward pass
            q_values_batch = self.session.run(self.q_values, feed_dict={self.state:new_state_batch})

            for i in range(0, self.batch_size):
                if chain_end_batch[i]:
                    q_value_fact_batch.append(reward_batch[i])
                else:
                    q_value_fact_batch.append(reward_batch[i] + self.gamma * np.max(q_values_batch[i]))
            
            #backward pass
            self.session.run(self.optimizer, feed_dict={
                self.state: state_batch,
                self.action: action_batch,
                self.q_value_fact: q_value_fact_batch
            })
            self.age += 1
        
    def decide(self, state, determistic=False):
        epsilon = 0
        if self.learning and (not determistic):
            epsilon = min(1.0, max(self.epsilon_min, 1.0-(self.age - self.learning_steps_burin)/(self.learning_steps_total - self.learning_steps_burin)))
        else:
            epsilon = self.epsilon_test_time
        
        action_index = 0
        if random.random() <= epsilon:
            action_index = random.randrange(0, self.num_actions)
        else:
            q_values = self.session.run(self.q_values, feed_dict={self.state:[state]})
            action_index = np.argmax(q_values)
        action = np.zeros(self.num_actions)      
        action[action_index] = 1     
            
        return action

    def _init_configs(self, config):
        self.num_actions = config.get('num_actions', 5)
        self.state_dimensions = config.get('state_dimensions', 250)
        self.experience_size = config.get('experience_size', 3000)
        self.start_learn_threshold = config.get('start_learn_threshold', 500)
        self.gamma = config.get('gamma', 0.7)
        self.learning_steps_total = config.get('learning_steps_total', 10000)
        self.learning_steps_burin = config.get('learning_steps_burin', 1000)
        self.epsilon_min = config.get('epsilon_min', 0.0)
        self.epsilon_test_time = config.get('epsilon_test_time', 0.0)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)

        print("num_actions:\t%d" % self.num_actions)
        print("state_dimensions:\t%d" % self.state_dimensions)
        print("experience_size:\t%d" % self.experience_size)
        print("start_learn_threshold:\t%d" % self.start_learn_threshold)
        print("gamma:\t%f" % self.gamma)
        print("learning_steps_total:\t%d" % self.learning_steps_total)
        print("learning_steps_burin:\t%d" % self.learning_steps_burin)
        print("epsilon:\t%f" % self.epsilon_min)
        print("epsilon_test_time:\t%f" % self.epsilon_test_time)
        print("learning_rate:\t%f" % self.learning_rate)
        print("batch_size:\t%d" % self.batch_size)
    
    def _build_network(self):
        self.state = tf.placeholder("float", [None, self.state_dimensions])
        self.action = tf.placeholder("float", [None, self.num_actions])
        self.q_value_fact = tf.placeholder("float", [None])
        
        num_neurons_fc1 = 64
        num_neurons_fc2 = 16
        
        w_fc1 = tf.Variable(tf.truncated_normal((self.state_dimensions, num_neurons_fc1)))
        b_fc1 = tf.Variable(tf.zeros(num_neurons_fc1))
        fc1 = tf.add(tf.matmul(self.state, w_fc1), b_fc1)
#        fc1 = tf.nn.relu(fc1)
#        keep_prob = tf.placeholder(tf.float32)
#        fc1 = tf.nn.dropout(fc1, keep_prob)

        w_fc2 = tf.Variable(tf.truncated_normal((num_neurons_fc1, num_neurons_fc2)))
        b_fc2 = tf.Variable(tf.zeros(num_neurons_fc2))
        fc2 = tf.add(tf.matmul(fc1, w_fc2), b_fc2)
#        fc2 = tf.nn.relu(fc2)
    
        w_fc3 = tf.Variable(tf.truncated_normal((num_neurons_fc2, self.num_actions)))
        b_fc3 = tf.Variable(tf.zeros(self.num_actions))
        self.q_values = tf.add(tf.matmul(fc2, w_fc3), b_fc3)
        
        q_value = tf.reduce_sum(tf.mul(self.q_values, self.action), reduction_indices = 1)

        cost = tf.reduce_mean(tf.square(self.q_value_fact - q_value))
        
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
    
    



