import time
import copy
import numpy as np
from config import VNFGroupConfig

group_config = VNFGroupConfig()

P = 10.0


class FLAG:
    def __init__(self):
        self.ALPHA = 1.0
        self.BETA = 2.5
        self.RHO = 0.5
        self.Q = 100.0
        self.ANT_NUM = 5
        self.ITER_MAX = 10
        self.VNF_PER_GROUP = 5
        self.VNF_PER_CHAIN = 5


flag = FLAG()


class Env:
    def __init__(self, B, D, B_, D_, pheromone_graph, distance_graph):
        self.B = B
        self.D = D
        self.B_ = B_
        self.D_ = D_
        self.pheromone_graph = pheromone_graph
        self.distance_graph = distance_graph


class Ant:
    def __init__(self, id, env):
        self.id = id
        self.env = env

    def __lt__(self, other):
        return self.total_distance < other.total_distance

    def __reset(self):
        self.path = []
        self.total_distance = 0.0
        self.current_node = (0, np.random.randint(0, flag.VNF_PER_GROUP))
        self.path.append(self.current_node)
        self.move_count = 1

    def __choose_next_node(self):
        next_node = [self.current_node[0]+1, -1]
        select_prob = [0.0 for _ in xrange(flag.VNF_PER_GROUP)]
        total_prob = 0.0

        for j in xrange(flag.VNF_PER_GROUP):
            m, i = self.current_node
            select_prob[j] = pow(self.env.pheromone_graph[j, i, m], flag.ALPHA) * pow(1.0/self.env.distance_graph[j, i, m], flag.BETA)
            total_prob += select_prob[j]

        if total_prob > 0.0:
            prob = np.random.uniform(0.0, total_prob)
            for i in xrange(flag.VNF_PER_GROUP):
                prob -= select_prob[i]
                if prob < 0.0:
                    next_node[1] = i
                    break

        if next_node[1] == -1:
            next_node[1] = np.random.randint(0, flag.VNF_PER_GROUP)

        return next_node

    def __move(self, next_node):
        self.path.append(next_node)
        self.total_distance += self.env.distance_graph[next_node[1], self.current_node[1], self.current_node[0]]
        self.current_node = next_node
        self.move_count += 1

    def __cal_total_distance(self):
        distance = 0.0
        for m in xrange(flag.VNF_PER_CHAIN-1):
            distance += self.env.distance_graph[self.path[m+1][1], self.path[m][1], m]
        self.total_distance = distance

    def __check_bandwidth(self, B_):
        for m in xrange(flag.VNF_PER_CHAIN-1):
            if self.env.B[self.path[m+1][1], self.path[m][1], m] < B_:
                return False
        return True

    def __check_delay(self, D_):
        d_sum = 0.0
        for m in xrange(flag.VNF_PER_CHAIN-1):
            d_sum += self.env.D[self.path[m+1][1], self.path[m][1], m]
        if d_sum > D_:
            return False
        return True

    def __check_is_valid(self):
        return self.__check_bandwidth(self.env.B_) and self.__check_delay(self.env.D_)

    def search_path(self):
        self.__reset()
        while self.move_count < flag.VNF_PER_CHAIN:
            next_node = self.__choose_next_node()
            self.__move(next_node)
        if self.__check_is_valid():
            self.__cal_total_distance()
        else:
            self.total_distance = np.inf


class SFC:
    def __init__(self, env):
        self.env = env
        self.ants = [Ant(i, self.env) for i in xrange(flag.ANT_NUM)]
        self.best_ant = Ant(-1, self.env)
        self.best_ant.total_distance = np.inf
        self.env.distance_graph = self.env.D.copy()

    def __update_pheromone_graph(self):
        pheromone_delta = np.zeros((flag.VNF_PER_GROUP, flag.VNF_PER_GROUP, flag.VNF_PER_CHAIN-1), np.float)
        for ant in self.ants:
            for m in xrange(flag.VNF_PER_CHAIN-1):
                pheromone_delta[ant.path[m+1][1], ant.path[m][1], m] += flag.Q/ant.total_distance
        self.env.pheromone_graph *= flag.RHO
        self.env.pheromone_graph += pheromone_delta

    def search_path(self):
        for _ in xrange(flag.ITER_MAX):
            for ant in self.ants:
                ant.search_path()
                if ant < self.best_ant:
                    self.best_ant = copy.deepcopy(ant)
            self.__update_pheromone_graph()


class ACO:
    def __init__(self):
        self.B = group_config.get_initialized_bandwidth()
        self.D = group_config.get_initialized_delay()
        self.sfc_requests = group_config.get_test_sfc()
        self.distance_graph = np.zeros((flag.VNF_PER_GROUP, flag.VNF_PER_GROUP, flag.VNF_PER_CHAIN-1), np.float)
        self.pheromone_graph = np.ones((flag.VNF_PER_GROUP, flag.VNF_PER_GROUP, flag.VNF_PER_CHAIN-1), np.float)

        self.running_sfc = np.ndarray([0, 1+flag.VNF_PER_CHAIN], dtype=np.int32)
        self.total_qoe = 0.0
        self.error_counter = 0

        self.env = None
        self.sfc = None

    def reset(self):
        self.__init__()

    def set_sfc_requests(self, sfc_requests):
        self.sfc_requests = sfc_requests

    def __random_release_sfc(self, thresh=0.2):
        prob = np.array([np.random.random() for _ in xrange(self.running_sfc.shape[0])])
        released_sfc = self.running_sfc[prob <= thresh]
        self.running_sfc = self.running_sfc[prob > thresh]
        for c in released_sfc:
            for i in xrange(flag.VNF_PER_CHAIN-1):
                self.B[c[i+2], c[i+1], i] += c[0]

    def __allocate_bandwidth(self, path, B_):
        for i in xrange(flag.VNF_PER_CHAIN-1):
            self.B[path[i+1][1], path[i][1], i] -= B_

    def __cal_qoe(self, path, B_, D_):
        d_sum = 0.0
        for m in xrange(flag.VNF_PER_CHAIN-1):
            d_sum += self.env.D[path[m+1][1], path[m][1], m]
        return np.log(B_) - P*np.exp(-(D_-d_sum)/10.0)

    def select(self):
        [B_, D_] = self.sfc_requests.pop(0)
        while True:
            self.env = Env(self.B, self.D, B_, D_, self.pheromone_graph, self.distance_graph)
            self.sfc = SFC(self.env)

            self.__random_release_sfc()

            self.sfc.search_path()

            if self.sfc.best_ant.total_distance != np.inf:
                path = self.sfc.best_ant.path
                self.total_qoe += self.__cal_qoe(path, B_, D_)
                self.__allocate_bandwidth(path, B_)
                sfc = [B_]
                sfc += np.array(path)[:, 1].tolist()
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
    aco = ACO()
    aco.set_sfc_requests(group_config.get_test_sfc())
    aco.select()
    print 'Mean QoE:', aco.get_mean_qoe()
    print 'Error Rate:', aco.get_error_rate()
