import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from collections import deque

class Experience(object):
    """Transition of (state, action, reward, next_state).
    The chain_end flag discripts whether next_state is the end of a Markov chain.
    
    """
    def __init__(self, state, action, reward, next_state, chain_end=False):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.chain_end = chain_end


class TFBrain(object):
    """Q-learning network reinforcement learning based on TensorFlow  """  
    
    def __init__(self, config):
        self._init_configs(config)
        if self.network_type == 'mlp':
            self._build_mlp_network()
        else:
            self._build_cnn_network()
        # Replay memory of experiences
        self.epsilon = 0.0
        self.experiences = deque()
        self.age = 0
        
    def _init_configs(self, config):
        self.network_type = config.get('network_type', 'mlp')
        self.learning = config.get('learning', True)
        self.num_actions = config.get('num_actions', 5)
        self.state_dimensions = config.get('state_dimensions', 250)
        self.experience_size = config.get('experience_size', 3000)
        self.observe_age = config.get('observe_age', 500)
        self.gamma = config.get('gamma', 0.7)
        self.explore_age = config.get('explore_age', 10000)
        self.explore_burin = config.get('explore_burin', 1000)
        self.epsilon_min = config.get('epsilon_min', 0.0)
        self.epsilon_test_time = config.get('epsilon_test_time', 0.0)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)
        self.lookback_window = config.get('lookback_window', 3)

    def _build_mlp_network(self):
        # MLP (multi-layer perceptron) network for Q-learning
        # For training, the sample is (state, action) pair, and the label is 'fact' Q-value Q(s, a) = r + gamma * max(Q(ss, aa)) 
        # For inferrence, the input is state, and the output of the network is the Q-values of all possible actions for the state.
        self.state = tf.placeholder("float", [None, self.state_dimensions])
        self.action = tf.placeholder("float", [None, self.num_actions])
        self.fq_value = tf.placeholder("float", [None])
        
        # 1st fully connected layer of 64 hidden units
        num_neurons_fc1 = 64
        w_fc1 = tf.Variable(tf.truncated_normal((self.state_dimensions, num_neurons_fc1)))
        b_fc1 = tf.Variable(tf.zeros(num_neurons_fc1))
        fc1   = tf.add(tf.matmul(self.state, w_fc1), b_fc1)

        # 2nd fully connected layer of 16 hidden units
        num_neurons_fc2 = 16
        w_fc2 = tf.Variable(tf.truncated_normal((num_neurons_fc1, num_neurons_fc2)))
        b_fc2 = tf.Variable(tf.zeros(num_neurons_fc2))
        fc2   = tf.add(tf.matmul(fc1, w_fc2), b_fc2)
    
        # 3rd fully connected layer of outputs (Q-values of all actions on the state)
        w_fc3 = tf.Variable(tf.truncated_normal((num_neurons_fc2, self.num_actions)))
        b_fc3 = tf.Variable(tf.zeros(self.num_actions))
        self.q_values = tf.add(tf.matmul(fc2, w_fc3), b_fc3)
        
        # Get the Q-value of the (state, action) pair
        q_value = tf.reduce_sum(tf.mul(self.q_values, self.action), reduction_indices = 1)
        # The optimization goal is to minimize the discrepancy between 'inferred' Q-value and the 'fact' Q-value
        cost = tf.reduce_mean(tf.square(self.fq_value - q_value))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
    
    def _build_cnn_network(self):
        # CNN (convolutional neural network) for Q-learning
        # For training, the sample is (state, action) pair, and the label is 'fact' Q-value Q(s, a) = r + gamma * max(Q(ss, aa)) 
        # For inferrence, the input is state, and the output of the network is the Q-values of all possible actions for the state.
        input_channels = self.lookback_window + 1
        self.state = tf.placeholder("float", [None, 80, 80, input_channels])
        self.action = tf.placeholder("float", [None, self.num_actions])
        self.fq_value = tf.placeholder("float", [None])

        sigma = 0.01
        conv1_W = tf.Variable(tf.truncated_normal(shape=(8, 8, input_channels, 32), stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(32))
        conv1   = tf.nn.conv2d(self.state, conv1_W, strides=[1, 4, 4, 1], padding='SAME') + conv1_b
        conv1   = tf.nn.relu(conv1)
        conv1   = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        conv2_W = tf.Variable(tf.truncated_normal(shape=(4, 4, 32, 64), stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(64))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 2, 2, 1], padding='SAME') + conv2_b
        conv2   = tf.nn.relu(conv2)

        conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), stddev = sigma))
        conv3_b = tf.Variable(tf.zeros(64))
        conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
        conv3   = tf.nn.relu(conv3)

        fc0     = flatten(conv3)
        fc1_W   = tf.Variable(tf.truncated_normal(shape=(1600, 512), stddev = sigma))
        fc1_b   = tf.Variable(tf.zeros(512))
        fc1     = tf.matmul(fc0, fc1_W) + fc1_b
        fc1     = tf.nn.relu(fc1)
        
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(512, self.num_actions), stddev = sigma))
        fc2_b  = tf.Variable(tf.zeros(self.num_actions))
        self.q_values = tf.matmul(fc1, fc2_W) + fc2_b

        # Get the Q-value of the (state, action) pair
        q_value = tf.reduce_sum(tf.mul(self.q_values, self.action), reduction_indices = 1)
        # The optimization goal is to minimize the discrepancy between 'inferred' Q-value and the 'fact' Q-value
        cost = tf.reduce_mean(tf.square(self.fq_value - q_value))
        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost) 
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
    
    def learn(self, experience):
        """ Q-learning network implemented by TensorFlow.
        @param experience: instance of Experience.
        @logic: for experience (s, a, r, ss), update Q(s, a) --> r + gamma * max(Q(ss, aa)) for all possible actions aa of ss

        """
        # Step 1: update replay memory

        self.experiences.append(experience)
        if len(self.experiences) > self.experience_size:
            self.experiences.popleft()

        if self.learning and self.age > self.observe_age and len(self.experiences) > self.batch_size:
            # Step 2: Sample training batch from replay memory
            batch = random.sample(self.experiences, self.batch_size)
            state_batch = [e.state for e in batch]
            action_batch = [e.action for e in batch]
            reward_batch = [e.reward for e in batch]
            next_state_batch = [e.next_state for e in batch]
            chain_end_batch = [e.chain_end for e in batch]
            fq_value_batch = []

            # Step 3: Forward pass of the Q-learning network to calculate the Q-values of next_state, saying Q(ss, aa)
            q_values_batch = self.session.run(self.q_values, feed_dict={self.state:next_state_batch})

            # Step 4: Update the Q-value of current state using the 'fact' of instant reward got from the experience
            for i in range(0, self.batch_size):
                # No further transition states when reaching the end of chain (end state of invalid state)
                if chain_end_batch[i]:
                    fq_value_batch.append(reward_batch[i])
                # Q(s, a) = r + gamma * max(Q(ss, aa)) for all possible actions aa of ss
                else:
                    fq_value_batch.append(reward_batch[i] + self.gamma * np.max(q_values_batch[i]))
            
            # Step 5: Backward pass of the Q-learning network, to refine weights by learning from the 'fact' of experience
            self.session.run(self.optimizer, feed_dict={
                self.state: state_batch,
                self.action: action_batch,
                self.fq_value: fq_value_batch
            })

        self.age += 1
        stage = ""
        if self.age <= self.observe_age:
            stage = "observe"
        elif self.age <= self.explore_age:
            stage = "explore"
        else:
            stage = "train"
        if self.network_type == 'cnn' and self.age % 100 == 0:
            print("ITERATIONS: ", self.age, "\tSTAGE: ", stage, "\tEPSILON: ", self.epsilon )

        if self.age % 10000 == 0:                                                                                                                                    
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.age)
        
    def decide(self, state, determistic=True):
        """ Predict the best action for the state by forwarding pass the network
        @param determistic - 'False' can ONLY be used for network training purpose. Choose 'True' when using an already trained network.

        """
        if not determistic:
            assert(self.learning is True)

        if self.learning and (not determistic):
            # Purely 'explore' before the learning has proceeded at least learning_steps_burin times.
            self.epsilon = min(1.0, max(self.epsilon_min, 1.0-(self.age - self.explore_burin)/(self.explore_age - self.explore_burin)))
        else:
            self.epsilon = self.epsilon_test_time

        action = np.zeros(self.num_actions)
        idx = 0
        if random.random() <= self.epsilon:
            idx = random.randrange(self.num_actions)
        else:
            # Forward pass the network to calculate Q-values for all actions and select the max one. 
            q_values = self.session.run(self.q_values, feed_dict={self.state:[state]})
            idx = np.argmax(q_values)
        action[idx] = 1     
            
        return action

    def show_configs(self):
        if self.network_type == 'mlp':
            print("-- network_type:\t%s" % self.network_type)
            print("-- num_actions:\t%d" % self.num_actions)
            print("-- state_dimensions:\t%d" % self.state_dimensions)
            print("-- experience_size:\t%d" % self.experience_size)
            print("-- observe_age:\t%d" % self.observe_age)
            print("-- gamma:\t%f" % self.gamma)
            print("-- explore_age:\t%d" % self.explore_age)
            print("-- explore_burin:\t%d" % self.explore_burin)
            print("-- epsilon_min:\t%f" % self.epsilon_min)
            print("-- epsilon_test_time:\t%f" % self.epsilon_test_time)
            print("-- learning_rate:\t%f" % self.learning_rate)
            print("-- batch_size:\t%d" % self.batch_size)
        else:
            print("-- network_type:\t%s" % self.network_type)
            print("-- num_actions:\t%d" % self.num_actions)
            print("-- experience_size:\t%d" % self.experience_size)
            print("-- observe_age:\t%d" % self.observe_age)
            print("-- gamma:\t%f" % self.gamma)
            print("-- explore_age:\t%d" % self.explore_age)
            print("-- explore_burin:\t%d" % self.explore_burin)
            print("-- epsilon_min:\t%f" % self.epsilon_min)
            print("-- epsilon_test_time:\t%f" % self.epsilon_test_time)
            print("-- batch_size:\t%d" % self.batch_size)
            print("-- lookback_window:\t%d" % self.lookback_window)
            
