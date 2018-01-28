from dqn import DQN
from env import VNFGroup


if __name__ == '__main__':
    env = VNFGroup()
    agent = DQN()
    agent.load()
    while True:
        env.reset()
        [B_, D_] = env.sfc_requests.pop(0)
        while True:
            observation = env.start(B_, D_)
            for n in xrange(5):
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                agent.store_transition(observation, action, reward, observation_)
                if done:  break
                observation = observation_
            agent.learn()
            try:
                [B_, D_] = env.sfc_requests.pop(0)
            except IndexError:
                break
        print env.get_mean_qoe()
        agent.save()
