"""
"""

import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import threading, queue
from CHP.CHP_MODEL import CHPEnv
import datetime


EP_MAX = 2500
EP_LEN = 300
N_WORKER = 4  # parallel workers
GAMMA = 0.9  # reward discount factor
A_LR = 0.000005  # learning rate for actor
C_LR = 0.00002  # learning rate for critic
MIN_BATCH_SIZE = 24  # minimum batch size for updating PPO
UPDATE_STEP = 5  # loop update operation n-steps
EPSILON = 0.2  # Clipped surrogate objective
ON_TRAIN = False

env = CHPEnv()
S_DIM = env.state_dim
A_DIM = env.action_dim
A_BOUND = env.action_bound[1]


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()

        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        w_init = tf.contrib.layers.xavier_initializer()
        l_c = tf.layers.dense(self.tfs, 300, tf.nn.relu, kernel_initializer=w_init, name='lc')
        l_c1 = tf.layers.dense(l_c, 100, tf.nn.relu, kernel_initializer=w_init, name='lc1')
        self.v = tf.layers.dense(l_c1, 1, kernel_initializer=w_init, name='v')  # state value
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        tf.summary.scalar('critic_loss', self.closs)
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=True)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv  # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))
        tf.summary.scalar('actor_loss', self.aloss)
        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.merged = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.sess.run(self.update_oldpi_op)  # old pi to pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 300, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            mu = A_BOUND * tf.layers.dense(l2, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l2, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -1, 1)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = CHPEnv()
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer
                a = self.ppo.choose_action(s)
                s_, r, done = self.env.step(a)
                if t == EP_LEN - 1: done = True
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r/100)  # normalize reward, find to be useful
                s = s_
                ep_r += r/100

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size
                if done or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= EP_MAX:  # stop training
                        COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )


if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    if ON_TRAIN:
        starttime = datetime.datetime.now()
        UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
        UPDATE_EVENT.clear()  # no update now
        ROLLING_EVENT.set()  # start to roll out
        workers = [Worker(wid=i) for i in range(N_WORKER)]

        GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
        GLOBAL_RUNNING_R = []
        COORD = tf.train.Coordinator()
        QUEUE = queue.Queue()
        threads = []
        for worker in workers:  # worker threads
            t = threading.Thread(target=worker.work, args=())
            t.start()
            threads.append(t)
        # add a PPO updating thread
        threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
        threads[-1].start()
        COORD.join(threads)
        endtime = datetime.datetime.now()
        print((endtime - starttime).seconds)
        GLOBAL_PPO.save()
        plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        np.savetxt('R_dppo.txt', GLOBAL_RUNNING_R, delimiter=',')
        plt.xlabel('Episode');
        plt.ylabel('Moving reward');
        plt.ion();
        plt.show()
    else:
        GLOBAL_PPO.restore()
        s = env.set()
        print(env.device_info)
        state = np.zeros((300, 14))
        device = np.zeros((300, 4, 4))
        dis_p = np.zeros(300)
        dis_q = np.zeros(300)
        cost = np.zeros(300)
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]
            state[t, :] = s
            cost[t] = env.realcost
            print(env.device_info)
            device[t, :, :] = env.device_info
            cost[t] = env.realcost
        np.save("state.npy", state)
        np.save("device.npy", device)
        np.save("cost.npy", cost)





