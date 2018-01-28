from env import VNFGroup
from dqn import DQN
from random_sfc import RandomSFC
from violent_sfc import ViolentSFC


if __name__ == '__main__':
    env = VNFGroup()
    env.reset()

    # Random
    random_sfc = RandomSFC()
    random_sfc.set_sfc_requests(env.sfc_requests[:])
    random_sfc.select()

    # Voilent
    violent_sfc = ViolentSFC()
    violent_sfc.set_sfc_requests(env.sfc_requests[:])
    violent_sfc.select()

    # DQN
    agent = DQN()
    agent.load()
    sfc_requests = env.sfc_requests[:]
    dqn_error = 0
    [B_, D_] = sfc_requests.pop(0)
    while True:
        observation = env.start(B_, D_)
        for n in xrange(5):
            action = agent.choose_action(observation, larger_greedy=1.0)
            observation_, reward, done, info = env.step(action)
            if info['id']:  dqn_error += 1
            if done:  break
            observation = observation_
        try:
            [B_, D_] = sfc_requests.pop(0)
        except IndexError:
            break
    print '{0}, {1}, {2}, {3}, {4}, {5}'.format(random_sfc.get_mean_qoe(), random_sfc.get_random_error(),
                                                violent_sfc.get_mean_qoe(), violent_sfc.get_violent_error(),
                                                env.get_mean_qoe(), dqn_error)
