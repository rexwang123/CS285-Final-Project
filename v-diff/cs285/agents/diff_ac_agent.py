from collections import OrderedDict

from cs285.critics.diff_bootstrapped_continuous_critic import \
    DiffBootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
from cs285.infrastructure import pytorch_util as ptu
import torch


class DiffACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(DiffACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = DiffBootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no1, ac_na1, re_n1, next_re_n1, next_ob_no1, terminal_n1, ob_no2, ac_na2, re_n2, next_ob_no2, terminal_n2, mt = True):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # advantage = estimate_advantage(...)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        loss = OrderedDict()
        loss['Critic_Loss'] = 0
        loss['Actor_Loss'] = 0

        num = min(len(ob_no1), len(ob_no2))
        if len(ob_no1) < len(ob_no2):
          ob_no2 = ob_no2[:num]
          next_ob_no2 = next_ob_no2[:num]
          ac_na2 = ac_na2[:num]
          terminal_n2 = terminal_n2[:num]
        else:
          ob_no1 = ob_no1[:num]
          ac_na1 = ac_na1[:num]
          next_ob_no1 = next_ob_no1[:num]
          terminal_n1 = terminal_n1[:num]

        print(len(re_n2),len(re_n1), len(terminal_n2), len(terminal_n1))
        if(mt == False):
            for i in range(self.agent_params['num_critic_updates_per_agent_update']):
                loss['Critic_Loss'] += self.critic.update(ob_no1,ac_na1,next_ob_no1, re_n1,terminal_n1,ob_no2,ac_na2,next_ob_no2,re_n2, terminal_n2, False)
        else:
            for i in range(self.agent_params['num_critic_updates_per_agent_update']):
                loss['Critic_Loss'] += self.critic.update(ob_no1,ac_na1,next_ob_no1, re_n1,terminal_n1,ob_no2,ac_na2,next_ob_no2,re_n2, terminal_n2, True)
        
        diff = self.estimate_diff(ob_no1, next_ob_no1, re_n1, next_re_n1, terminal_n1)
        self.actor.update(ob_no1, ac_na1, diff)
        
        for i in range(self.agent_params['num_actor_updates_per_agent_update']):
            loss['Actor_Loss'] += self.actor.update(ob_no1,ac_na1,diff)
        
        return loss
       

    def estimate_diff(self, ob_no, next_ob_no, re_n, next_re_n, terminal_n):
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)
        re_n = ptu.from_numpy(re_n)
        next_re_n = ptu.from_numpy(next_re_n)
        # re_n = ptu.from_numpy(re_n)
        # next_re_n = ptu.from_numpy(next_re_n)
        
        num = len(ob_no)
        re_n = re_n[:num]
        combine = torch.cat((next_ob_no, ob_no), -1)
        diff = re_n + self.gamma * self.critic(combine) * (1-terminal_n)
        # diff = re_n + self.gamma * self.critic(ob_no) * (1-terminal_n)
        diff = ptu.to_numpy(diff)
        if self.standardize_advantages:
            diff = (diff - np.mean(diff)) / (np.std(diff) + 1e-8)
        
        return diff
        

    # def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
    #     # TODO Implement the following pseudocode:
    #     # 1) query the critic with ob_no, to get V(s)
    #     # 2) query the critic with next_ob_no, to get V(s')
    #     # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
    #     # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
    #     # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
    #     ob_no = ptu.from_numpy(ob_no)
    #     next_ob_no = ptu.from_numpy(next_ob_no)
    #     re_n = ptu.from_numpy(re_n)
    #     terminal_n = ptu.from_numpy(terminal_n)

    #     v = self.critic(ob_no)
    #     v_ = self.critic(next_ob_no) * (1-terminal_n)
    #     q = re_n + self.gamma * v_ 
    #     adv_n = q - v
    #     adv_n = ptu.to_numpy(adv_n)
    #     if self.standardize_advantages:
    #         adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
    #     return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
