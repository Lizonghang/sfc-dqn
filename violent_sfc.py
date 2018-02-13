# coding=utf-8
from config import VNFGroupConfig
import numpy as np
import time

group_config = VNFGroupConfig()

P = 10.0


class ViolentSFC:
    def __init__(self):
        self.B = group_config.get_initialized_bandwidth()
        self.D = group_config.get_initialized_delay()
        self.sfc_requests = group_config.get_test_sfc()
        self.running_sfc = np.ndarray([0, 6], dtype=np.int32)
        self.total_qoe = 0.0
        self.error_counter = 0

    def reset(self):
        self.__init__()

    def set_sfc_requests(self, sfc_requests):
        self.sfc_requests = sfc_requests

    def random_release_sfc(self, thresh=0.2):
        prob = np.array([np.random.random() for _ in xrange(self.running_sfc.shape[0])])
        released_sfc = self.running_sfc[prob <= thresh]
        self.running_sfc = self.running_sfc[prob > thresh]
        for c in released_sfc:
            for i in xrange(0, 4):
                self.B[c[i+2], c[i+1], i] += c[0]

    def check_B(self, c, B_):
        for vnf_id in xrange(1, 5):
            if self.B[c[vnf_id], c[vnf_id-1], vnf_id-1] < B_:
                return False
        return True

    def check_D(self, c, D_):
        d_sum = 0.0
        for vnf_id in xrange(1, 5):
            d_sum += self.D[c[vnf_id], c[vnf_id-1], vnf_id-1]
        if d_sum > D_:
            return 0, False
        return d_sum, True

    def allocate_B(self, c, B_):
        for vnf_id in xrange(1, 5):
            self.B[c[vnf_id], c[vnf_id-1], vnf_id-1] -= B_

    def select(self):
        [B_, D_] = self.sfc_requests.pop(0)
        while True:
            self.random_release_sfc()
            # start_time = time.time()
            best_c = None
            best_qoe = 0.0
            for node1 in xrange(5):
                for node2 in xrange(5):
                    for node3 in xrange(5):
                        for node4 in xrange(5):
                            for node5 in xrange(5):
                                c = [node1, node2, node3, node4, node5]
                                if self.check_B(c, B_):
                                    d_sum, flag = self.check_D(c, D_)
                                    if flag:
                                        qoe = np.log(B_) - P*np.exp(-(D_-d_sum)/10.0)
                                        if qoe > best_qoe:
                                            best_qoe = qoe
                                            best_c = c
            # print 'Time:', time.time() - start_time
            if best_c:
                self.total_qoe += best_qoe
                self.allocate_B(best_c, B_)
                sfc = [B_]
                sfc += best_c
                self.running_sfc = np.concatenate([self.running_sfc, np.array([sfc])], axis=0)
            else:
                self.total_qoe -= P
                self.error_counter += 1
            try:
                [B_, D_] = self.sfc_requests.pop(0)
            except IndexError:
                break

    def get_mean_qoe(self):
        return self.total_qoe / 100.0

    def get_error_rate(self):
        return self.error_counter / 100.0


if __name__ == '__main__':
    sfc = ViolentSFC()
    sfc.select()
    print 'Mean QoE:', sfc.get_mean_qoe()
    print 'Error Rate:', sfc.get_error_rate()
