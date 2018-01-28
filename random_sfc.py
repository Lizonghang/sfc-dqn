# coding=utf-8
from config import VNFGroupConfig
import numpy as np
import time

group_config = VNFGroupConfig()


class RandomSFC:
    def __init__(self):
        self.B = group_config.get_initialized_bandwidth()
        self.D = group_config.get_initialized_delay()
        self.sfc_requests = group_config.get_test_sfc()
        self.running_sfc = np.ndarray([0, 6], dtype=np.int32)
        self.total_qoe = 0.0
        self.error_counter = 0

    def set_sfc_requests(self, sfc_requests):
        self.sfc_requests = sfc_requests

    def random_release_sfc(self, thresh=0.2):
        prob = np.array([np.random.random() for _ in xrange(self.running_sfc.shape[0])])
        released_sfc = self.running_sfc[prob <= thresh]
        self.running_sfc = self.running_sfc[prob > thresh]
        for c in released_sfc:
            for i in xrange(0, 4):
                self.B[c[i+2], c[i+1], i] += c[0]

    def select(self):
        [B_, D_] = self.sfc_requests.pop(0)
        while True:
            # 随机释放
            self.random_release_sfc()
            # 随机选择一条链
            c = [np.random.choice(5) for _ in xrange(5)]
            # 标识为满足要求
            flag = True
            # 检查带宽是否满足要求
            if flag:
                for vnf_id in xrange(1, 5):
                    if self.B[c[vnf_id], c[vnf_id-1], vnf_id-1] < B_:
                        self.total_qoe -= 10
                        flag = False
                        break
            # 检查时延是否满足要求
            d_sum = 0.0
            if flag:
                for vnf_id in xrange(1, 5):
                    d_sum += self.D[c[vnf_id], c[vnf_id-1], vnf_id-1]
                if d_sum > D_:
                    self.total_qoe -= 10
                    flag = False
            # 分配成功
            if flag:
                # 累和qoe
                self.total_qoe += (100.0 / d_sum)
                # 分配带宽
                for vnf_id in xrange(1, 5):
                    self.B[c[vnf_id], c[vnf_id-1], vnf_id-1] -= B_
                # 记录资源分配
                sfc = [B_]
                sfc += c
                self.running_sfc = np.concatenate([self.running_sfc, np.array([sfc])], axis=0)
            else:
                self.error_counter += 1
            try:
                [B_, D_] = self.sfc_requests.pop(0)
            except IndexError:
                break

    def get_mean_qoe(self):
        return self.total_qoe / 100

    def get_random_error(self):
        return self.error_counter


if __name__ == '__main__':
    sfc = RandomSFC()
    sfc.select()
    print sfc.get_mean_qoe()
    print sfc.get_random_error()