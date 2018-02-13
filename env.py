# coding=utf-8
import numpy as np
import time
from config import VNFGroupConfig

group_config = VNFGroupConfig()

P = 10.0


class VNFGroup:
    def __init__(self):
        self.B = None            # B: 可用带宽,10-20G
        self.D = None            # D: 链路时延,10-20ms
        self.S = None            # B, B_, D, D_, Dc, action_index

        self.num_requests = 100  # 一次episode训练接受的sfc请求数
        self.B_min = 16          # 所需带宽范围最小值,单位MB
        self.B_max = 256         # 所需带宽范围最大值,单位MB
        self.D_min = 50          # 最大时延约束最小值,单位ms,>40ms
        self.D_max = 90          # 最大时延约束最大值,单位ms

        self.sfc_requests = None
        self.running_sfc = np.ndarray([0, 6], dtype=np.int32)  # [[B_,1,4,3,2,0]]
        self.c = []
        self.d_sum = 0.0

        self.total_qoe = 0.0

    def reset(self, use_sfc_requests=None):
        self.__init__()
        self.B = group_config.get_initialized_bandwidth()
        self.D = group_config.get_initialized_delay()
        self.S = None
        if use_sfc_requests:
            self.sfc_requests = use_sfc_requests
        else:
            self.sfc_requests = [[np.random.randint(self.B_min, self.B_max+1),
                                  np.random.randint(self.D_min, self.D_max+1)] for _ in xrange(self.num_requests)]

    def start(self, B_, D_):
        self.B_ = B_
        self.D_ = D_
        self.c = []
        self.d_sum = 0.0
        self.random_release_sfc()
        self.S = np.concatenate([self.B,                        # 0-3
                                 np.ones([5, 5, 1])*self.B_,    # 4
                                 self.D,                        # 5-8
                                 np.ones([5, 5, 1])*self.D_,    # 9
                                 np.ones([5, 5, 1])*self.d_sum, # 10
                                 np.ones((5, 5, 1)),            # 11
                                 np.zeros([5, 5, 1]),           # 12
                                 np.zeros([5, 5, 1]),           # 13
                                 np.zeros([5, 5, 1]),           # 14
                                 np.zeros([5, 5, 1])], axis=2)  # 15
        return self.S

    def random_release_sfc(self, thresh=0.2):
        prob = np.array([np.random.random() for _ in xrange(self.running_sfc.shape[0])])
        released_sfc = self.running_sfc[prob <= thresh]
        self.running_sfc = self.running_sfc[prob > thresh]
        for c in released_sfc:
            for i in xrange(4):
                self.B[c[i+2], c[i+1], i] += c[0]

    def allocate_bandwidth(self, c, B_):
        for i in xrange(4):
            self.B[c[i+1], c[i], i] -= B_

    def step(self, action):
        vnf_id = np.argmax([self.S[..., 11][0, 0],
                            self.S[..., 12][0, 0],
                            self.S[..., 13][0, 0],
                            self.S[..., 14][0, 0],
                            self.S[..., 15][0, 0]])
        if vnf_id == 0:
            self.c.append(action)
            self.S = np.concatenate([self.B,                        # 0-3
                                     np.ones([5, 5, 1])*self.B_,    # 4
                                     self.D,                        # 5-8
                                     np.ones([5, 5, 1])*self.D_,    # 9
                                     np.ones([5, 5, 1])*self.d_sum, # 10
                                     np.zeros((5, 5, 1)),           # 11
                                     np.ones([5, 5, 1]),            # 12
                                     np.zeros([5, 5, 1]),           # 13
                                     np.zeros([5, 5, 1]),           # 14
                                     np.zeros([5, 5, 1])], axis=2)  # 15
            reward = 0
            done = False
            info = {'id': 0, 'msg': 'SUCCESS: Choose node {} from VNF{}.'.format(action, vnf_id)}
            return self.S, reward, done, info
        elif vnf_id in (1, 2, 3):
            if self.B[action, self.c[-1], vnf_id-1] < self.B_:
                self.S = np.concatenate([self.B,                        # 0-3
                                         np.ones([5, 5, 1])*self.B_,    # 4
                                         self.D,                        # 5-8
                                         np.ones([5, 5, 1])*self.D_,    # 9
                                         np.ones([5, 5, 1])*self.d_sum, # 10
                                         np.zeros((5, 5, 1)),           # 11
                                         np.zeros([5, 5, 1]),           # 12
                                         np.zeros([5, 5, 1]),           # 13
                                         np.zeros([5, 5, 1]),           # 14
                                         np.zeros([5, 5, 1])], axis=2)  # 15
                reward = -P
                done = True
                info = {'id': 1, 'msg': 'FAIL: Bandwidth not enough.'}
                self.total_qoe += reward
                return self.S, reward, done, info

            self.d_sum += self.D[action, self.c[-1], vnf_id-1]
            if self.d_sum > self.D_:
                self.S = np.concatenate([self.B,                        # 0-3
                                         np.ones([5, 5, 1])*self.B_,    # 4
                                         self.D,                        # 5-8
                                         np.ones([5, 5, 1])*self.D_,    # 9
                                         np.ones([5, 5, 1])*self.d_sum, # 10
                                         np.zeros((5, 5, 1)),           # 11
                                         np.zeros([5, 5, 1]),           # 12
                                         np.zeros([5, 5, 1]),           # 13
                                         np.zeros([5, 5, 1]),           # 14
                                         np.zeros([5, 5, 1])], axis=2)  # 15
                reward = -P
                done = True
                info = {'id': 2, 'msg': 'FAIL: Delay over constraint'}
                self.total_qoe += reward
                return self.S, reward, done, info

            self.c.append(action)
            self.S = np.concatenate([self.B,                        # 0-3
                                     np.ones([5, 5, 1])*self.B_,    # 4
                                     self.D,                        # 5-8
                                     np.ones([5, 5, 1])*self.D_,    # 9
                                     np.ones([5, 5, 1])*self.d_sum, # 10
                                     np.zeros((5, 5, 1)),           # 11
                                     np.zeros([5, 5, 1]),           # 12
                                     np.zeros([5, 5, 1]),           # 13
                                     np.zeros([5, 5, 1]),           # 14
                                     np.zeros([5, 5, 1])], axis=2)  # 15
            self.S[..., 12+vnf_id] = np.ones([5, 5])
            reward = 0
            done = False
            info = {'id': 0, 'msg': 'SUCCESS: Choose node {} from VNF{}.'.format(action, vnf_id)}
            return self.S, reward, done, info
        else:
            if self.B[action, self.c[-1], vnf_id-1] < self.B_:
                self.S = np.concatenate([self.B,                        # 0-3
                                         np.ones([5, 5, 1])*self.B_,    # 4
                                         self.D,                        # 5-8
                                         np.ones([5, 5, 1])*self.D_,    # 9
                                         np.ones([5, 5, 1])*self.d_sum, # 10
                                         np.zeros((5, 5, 1)),           # 11
                                         np.zeros([5, 5, 1]),           # 12
                                         np.zeros([5, 5, 1]),           # 13
                                         np.zeros([5, 5, 1]),           # 14
                                         np.zeros([5, 5, 1])], axis=2)  # 15
                reward = -P
                done = True
                info = {'id': 1, 'msg': 'FAIL: Bandwidth not enough.'}
                self.total_qoe += reward
                return self.S, reward, done, info

            self.d_sum += self.D[action, self.c[-1], vnf_id-1]
            if self.d_sum > self.D_:
                self.S = np.concatenate([self.B,                        # 0-3
                                         np.ones([5, 5, 1])*self.B_,    # 4
                                         self.D,                        # 5-8
                                         np.ones([5, 5, 1])*self.D_,    # 9
                                         np.ones([5, 5, 1])*self.d_sum, # 10
                                         np.zeros((5, 5, 1)),           # 11
                                         np.zeros([5, 5, 1]),           # 12
                                         np.zeros([5, 5, 1]),           # 13
                                         np.zeros([5, 5, 1]),           # 14
                                         np.zeros([5, 5, 1])], axis=2)  # 15
                reward = -P
                done = True
                info = {'id': 2, 'msg': 'FAIL: Delay over constraint'}
                self.total_qoe += reward
                return self.S, reward, done, info

            self.c.append(action)
            # 添加到待释放SFC列表
            sfc = [self.B_]
            sfc += self.c
            self.running_sfc = np.concatenate([self.running_sfc, np.array([sfc])], axis=0)
            # 分配带宽资源
            self.allocate_bandwidth(self.c, self.B_)
            self.S = np.concatenate([self.B,                        # 0-3
                                     np.ones([5, 5, 1])*self.B_,    # 4
                                     self.D,                        # 5-8
                                     np.ones([5, 5, 1])*self.D_,    # 9
                                     np.ones([5, 5, 1])*self.d_sum, # 10
                                     np.zeros((5, 5, 1)),           # 11
                                     np.zeros([5, 5, 1]),           # 12
                                     np.zeros([5, 5, 1]),           # 13
                                     np.zeros([5, 5, 1]),           # 14
                                     np.zeros([5, 5, 1])], axis=2)  # 15
            reward = np.log(self.B_) - P*np.exp(-(self.D_-self.d_sum)/10.0)
            done = True
            info = {'id': 0, 'msg': 'SUCCESS: Choose node {} from VNF{}. Complete a SFC request.'.format(action, vnf_id)}
            self.total_qoe += reward
            return self.S, reward, done, info

    def get_mean_qoe(self):
        return self.total_qoe / self.num_requests

if __name__ == '__main__':
    env = VNFGroup()
    env.reset()
    print env.sfc_requests
