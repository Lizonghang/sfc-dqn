import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self,
                 lr=1e-3,
                 reward_decay=0.9,
                 e_greedy=0.8,
                 replace_target_iter=100,
                 memory_size=512,
                 batch_size=64
                 ):
        self.lr = lr
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, 5*5*16*2+2))
        self.build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def reshape_value(self, s):
        """
        B: 768-1280M
        B_:16-256M
        D: 10-20ms
        D_:60-100ms
        d_sum:40-80ms
        """
        B = (s[..., :4] - 768.0) / 512.0
        B_ = (s[..., 4:5] - 16.0) / 240.0
        D = (s[..., 5:9] - 10.0) / 10.0
        D_ = (s[..., 9:10] - 60.0) / 40.0
        d_sum = (s[..., 10:11] - 40.0) / 40.0
        other = s[..., 11:]
        return tf.concat([B, B_, D, D_, d_sum, other], axis=3)

    def build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, 5, 5, 16], name='s')
        self.s_reshaped = self.reshape_value(self.s)
        self.q_target = tf.placeholder(tf.float32, [None, 5], name='q_target')

        w_initializer = tf.random_normal_initializer(stddev=0.01)
        b_initializer = tf.constant_initializer(0.0)

        with tf.variable_scope('eval_net'):
            with tf.variable_scope('conv_net1'):
                w1 = tf.get_variable('w1', [3, 3, 16, 32],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                b1 = tf.get_variable('b1', [32],
                                     initializer=b_initializer,
                                     collections=['eval_net', 'variables'])
                conv1 = tf.nn.conv2d(self.s_reshaped, w1, strides=[1, 1, 1, 1], padding='VALID')
                h1 = tf.nn.relu(conv1 + b1)

            with tf.variable_scope('conv_net2'):
                w2 = tf.get_variable('w2', [3, 3, 32, 64],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                b2 = tf.get_variable('b2', [64],
                                     initializer=b_initializer,
                                     collections=['eval_net', 'variables'])
                conv2 = tf.nn.conv2d(h1, w2, strides=[1, 1, 1, 1], padding='VALID')
                h2 = tf.reshape(tf.nn.relu(conv2 + b2), [-1, 64])

            with tf.variable_scope('fc_net1'):
                w3 = tf.get_variable('w3', [64, 32],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                b3 = tf.get_variable('b3', [32],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

            with tf.variable_scope('fc_net2'):
                w4 = tf.get_variable('w4', [32, 5],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                b4 = tf.get_variable('b4', [5],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                self.q_eval = tf.matmul(h3, w4) + b4

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, 5, 5, 16], name='s_')
        self.s_reshaped_ = self.reshape_value(self.s_)

        with tf.variable_scope('target_net'):
            with tf.variable_scope('conv_net1'):
                w1 = tf.get_variable('w1', [3, 3, 16, 32],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                b1 = tf.get_variable('b1', [32],
                                     initializer=b_initializer,
                                     collections=['target_net', 'variables'])
                conv1 = tf.nn.conv2d(self.s_reshaped_, w1, strides=[1, 1, 1, 1], padding='VALID')
                h1 = tf.nn.relu(conv1 + b1)

            with tf.variable_scope('conv_net2'):
                w2 = tf.get_variable('w2', [3, 3, 32, 64],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                b2 = tf.get_variable('b2', [64],
                                     initializer=b_initializer,
                                     collections=['target_net', 'variables'])
                conv2 = tf.nn.conv2d(h1, w2, strides=[1, 1, 1, 1], padding='VALID')
                h2 = tf.reshape(tf.nn.relu(conv2 + b2), [-1, 64])

            with tf.variable_scope('fc_net1'):
                w3 = tf.get_variable('w3', [64, 32],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                b3 = tf.get_variable('b3', [32],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

            with tf.variable_scope('fc_net2'):
                w4 = tf.get_variable('w4', [32, 5],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                b4 = tf.get_variable('b4', [5],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                self.q_next = tf.matmul(h3, w4) + b4

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = np.hstack((s.reshape((400,)), [a, r], s_.reshape((400,))))
        self.memory_counter += 1

    def choose_action(self, observation, larger_greedy=0.0):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < max(self.e_greedy, larger_greedy):
            action_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, 5)
        return action

    def replace_target_net_params(self):
        target_net_params = tf.get_collection('target_net')
        eval_net_params = tf.get_collection('eval_net')
        self.sess.run([tf.assign(t, e)
                       for t, e in zip(target_net_params, eval_net_params)])

    def learn(self):
        # Check to replace target net params
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_net_params()

        # Sample batch memory
        batch_index = np.random.choice(min(self.memory_size, self.memory_counter), size=self.batch_size)
        batch_memory = self.memory[batch_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s: batch_memory[:, :5*5*16].reshape([self.batch_size, 5, 5, 16]),
                self.s_: batch_memory[:, -5*5*16:].reshape([self.batch_size, 5, 5, 16])
            })

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        actions = batch_memory[:, 5*5*16].astype(np.int32)
        rewards = batch_memory[:, 5*5*16+1]
        q_target[batch_index, actions] = rewards + self.reward_decay * np.max(q_next, axis=1)

        # Train eval net
        self.sess.run(self.train_op, feed_dict={self.s: batch_memory[:, :5*5*16].reshape([self.batch_size, 5, 5, 16]),
                                                self.q_target: q_target})

        self.learn_step_counter += 1

    def save(self, ckpt_file='ckpt/dqn.ckpt'):
        if not os.path.exists(os.path.dirname(ckpt_file)):
            os.makedirs(os.path.dirname(ckpt_file))
        self.saver.save(self.sess, ckpt_file)

    def load(self, ckpt_dir='ckpt'):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(ckpt_dir, ckpt_name))
            # print '[SUCCESS] Checkpoint loaded.'
        else:
            print '[ERROR] No checkpoint found.'


if __name__ == '__main__':
    agent = DQN()
