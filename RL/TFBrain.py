import random
import numpy as np
import tensorflow as tf


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
    
    def __init__(self, config, learning=True):    
        self._init_configs(config)
        self._build_network()
        # Replay memory of experiences
        self.experiences = []
        # Effective learning steps occurred so far, to control explore vs. exploit
        self.age = 0
        # Learn model from the scratch or load existing model
        self.learning = learning

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

    def _build_network(self):
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
        fc1 = tf.add(tf.matmul(self.state, w_fc1), b_fc1)

        # 2nd fully connected layer of 16 hidden units
        num_neurons_fc2 = 16
        w_fc2 = tf.Variable(tf.truncated_normal((num_neurons_fc1, num_neurons_fc2)))
        b_fc2 = tf.Variable(tf.zeros(num_neurons_fc2))
        fc2 = tf.add(tf.matmul(fc1, w_fc2), b_fc2)
    
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
    
    def learn(self, experience):
        """ Q-learning network implemented by TensorFlow.
        @param experience: instance of Experience.
        @logic: for experience (s, a, r, ss), update Q(s, a) --> r + gamma * max(Q(ss, aa)) for all possible actions aa of ss

        """

        # Step 1: update replay memory
        if len(self.experiences) < self.experience_size:
            self.experiences.append(experience)
        else:
            replace_index = random.randrange(self.experience_size)
            self.experiences[replace_index] = experience

        if(len(self.experiences) > max(self.start_learn_threshold, self.batch_size)):
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
        
    def decide(self, state, determistic=True):
        """ Predict the best action for the state by forwarding pass the network
        @param determistic - 'False' can ONLY be used for network training purpose. Choose 'True' when using an already trained network.

        """
        if not determistic:
            assert(self.learning is True)

        epsilon = 0
        if self.learning and (not determistic):
            # Purely 'explore' before the learning has proceeded at least learning_steps_burin times.
            epsilon = min(1.0, max(self.epsilon_min, 1.0-(self.age - self.learning_steps_burin)/(self.learning_steps_total - self.learning_steps_burin)))
        else:
            epsilon = self.epsilon_test_time

        idx = 0
        if random.random() <= epsilon:
            idx = random.randrange(0, self.num_actions)
        else:
            # Forward pass the network to calculate Q-values for all actions and select the max one. 
            q_values = self.session.run(self.q_values, feed_dict={self.state:[state]})
            idx = np.argmax(q_values)
        action = np.zeros(self.num_actions)      
        action[idx] = 1     
            
        return action

    def show_configs(self):
        print("-- num_actions:\t%d" % self.num_actions)
        print("-- state_dimensions:\t%d" % self.state_dimensions)
        print("-- experience_size:\t%d" % self.experience_size)
        print("-- start_learn_threshold:\t%d" % self.start_learn_threshold)
        print("-- gamma:\t%f" % self.gamma)
        print("-- learning_steps_total:\t%d" % self.learning_steps_total)
        print("-- learning_steps_burin:\t%d" % self.learning_steps_burin)
        print("-- epsilon:\t%f" % self.epsilon_min)
        print("-- epsilon_test_time:\t%f" % self.epsilon_test_time)
        print("-- learning_rate:\t%f" % self.learning_rate)
        print("-- batch_size:\t%d" % self.batch_size)
