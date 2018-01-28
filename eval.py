from dqn import DQN
from env import VNFGroup
import time


if __name__ == '__main__':
    env = VNFGroup()
    agent = DQN()
    agent.load()
    env.reset()
    error_counter = 0
    [B_, D_] = env.sfc_requests.pop(0)
    while True:
        # start_time = time.time()
        observation = env.start(B_, D_)
        for n in xrange(5):
            action = agent.choose_action(observation, larger_greedy=1.0)
            observation_, reward, done, info = env.step(action)
            if info['id']:  error_counter += 1
            if done:  break
            observation = observation_
        # print 'Time:', time.time() - start_time
        try:
            [B_, D_] = env.sfc_requests.pop(0)
        except IndexError:
            break
    print 'Mean QoE:', env.get_mean_qoe()
    print 'Error Count:', float(error_counter) / env.num_requests
