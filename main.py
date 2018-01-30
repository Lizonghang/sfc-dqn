import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from env import VNFGroup
from dqn import DQN
from random_sfc import RandomSFC
from violent_sfc import ViolentSFC


if __name__ == '__main__':
    env = VNFGroup()
    agent = DQN()
    agent.load()
    random_sfc = RandomSFC()
    violent_sfc = ViolentSFC()

    while True:
        env.reset()
        random_sfc.reset()
        violent_sfc.reset()
        # Random
        random_sfc.set_sfc_requests(env.sfc_requests[:])
        random_sfc.select()
        # Voilent
        violent_sfc.set_sfc_requests(env.sfc_requests[:])
        violent_sfc.select()
        # DQN
        sfc_requests = env.sfc_requests[:]
        dqn_error = 0.0
        [B_, D_] = sfc_requests.pop(0)
        while True:
            observation = env.start(B_, D_)
            for n in xrange(5):
                action = agent.choose_action(observation, larger_greedy=1.0)
                observation_, reward, done, info = env.step(action)
                agent.store_transition(observation, action, reward, observation_)
                if info['id']:  dqn_error += 1
                if done:  break
                observation = observation_
            agent.learn()
            try:
                [B_, D_] = sfc_requests.pop(0)
            except IndexError:
                break
        agent.save()
        with open('output.txt', 'a') as f:
            f.write('{0}, {1}, {2}, {3}, {4}, {5}\n'.format(
                random_sfc.get_mean_qoe(), random_sfc.get_error_rate(),
                violent_sfc.get_mean_qoe(), violent_sfc.get_error_rate(),
                env.get_mean_qoe(), dqn_error/env.num_requests)
            )
