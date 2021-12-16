from .base_critic import BaseCritic
from torch import nn
from torch import optim
import torch
import numpy as np

from cs285.infrastructure import pytorch_util as ptu


class DiffBootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.critic_network = ptu.build_mlp(
            2 * self.ob_dim + 2 * self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def forward(self, obs):
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        return ptu.to_numpy(predictions)

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        # TODO: create `list_of_discounted_returns`
        # HINT: it is possible to write a vectorized solution, but a solution
            # using a for loop is also fine

        list_of_discounted_cumsums = [np.sum([rewards[j] * self.gamma ** (j - i) for j in range(i, len(rewards))]) for i in range(len(rewards))]
        return list_of_discounted_cumsums

    def calculate_q_vals(self, rewards_list):
        q_values = np.concatenate([self._discounted_cumsum(r) for r in rewards_list])

        return q_values
    def update(self, ob_no1, ac_na1, next_ob_no1, reward_n1, terminal_n1, ob_no2, ac_na2, next_ob_no2, reward_n2, terminal_n2, actor, mt = True):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward
        
        ob_no1 = ptu.from_numpy(ob_no1)
        next_ob_no1 = ptu.from_numpy(next_ob_no1)
        ac_na1 = ptu.from_numpy(ac_na1)
        terminal_n1 = ptu.from_numpy(terminal_n1)

        ob_no2 = ptu.from_numpy(ob_no2)
        next_ob_no2 = ptu.from_numpy(next_ob_no2)
        ac_na2 = ptu.from_numpy(ac_na2)
        terminal_n2 = ptu.from_numpy(terminal_n2)

        num = len(ob_no1)
        if(mt == True):
            print(len(reward_n1))
            rew1 = self.calculate_q_vals(reward_n1)
            rew2 = self.calculate_q_vals(reward_n2)
            rew1 = rew1[:num]
            rew2 = rew2[:num]
            target = ptu.from_numpy(rew2 - rew1)
            target2 = ptu.from_numpy(rew1 - rew2)
        else:
            reward_n1 = ptu.from_numpy(np.concatenate(reward_n1))
            reward_n2 = ptu.from_numpy(np.concatenate(reward_n2))
            reward_n1 = reward_n1[:num]
            reward_n2 = reward_n2[:num]
            # reward_n1 = ptu.from_numpy(reward_n1)
            # reward_n2 = ptu.from_numpy(reward_n2)

        loss = 0

        sample_num = 100
        for k in range(10):
            if k == 0:
              indices = torch.arange(0, len(ob_no2), dtype=torch.long)
            else:
              indices = torch.randperm(len(ob_no2))
            ac_na2_ = ac_na2[indices]
            ob_no2_ = ob_no2[indices]
            next_ob_no2_ = next_ob_no2[indices]
            reward_n2_ = reward_n2[indices]
            terminal_n2_ = terminal_n2[indices]

            for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
                if i % self.num_grad_steps_per_target_update == 0:
                    if mt == False:
                        target = 0
                        target2 = 0
                        
                        for j in range(sample_num):
                          ac1 = actor.get_action(ptu.to_numpy(next_ob_no1))
                          ac2 = actor.get_action(ptu.to_numpy(next_ob_no2_))
                        
                          target += (reward_n2_ - reward_n1 + self.gamma * self(torch.cat((next_ob_no2_, ptu.from_numpy(ac2), next_ob_no1, ptu.from_numpy(ac1)), -1)) * (1 - terminal_n1 * terminal_n2_)).detach()
                          target2 += (reward_n1 - reward_n2_ + self.gamma * self(torch.cat((next_ob_no1, ptu.from_numpy(ac1), next_ob_no2_, ptu.from_numpy(ac2)), -1)) * (1 - terminal_n1 * terminal_n2_)).detach()
                        target /= sample_num
                        target2 /= sample_num
                        # v = self(ob_no1)
                        # target = (reward_n1 + self.gamma * v * (1-terminal_n1)).detach()
                # print(ob_no2.size(), ac_na2.size(), ob_no1.size(), ac_na1.size())
              
                curr = self(torch.cat((ob_no2_, ac_na2_, ob_no1, ac_na1), -1))
                curr2 = self(torch.cat((ob_no1, ac_na1, ob_no2_, ac_na2_), -1))
                curr3 = self(torch.cat((ob_no1, ac_na1, ob_no1, ac_na1), -1))
                l = self.loss(curr, target) + self.loss(curr2, target2) + 0.1 * self.loss(curr3,torch.zeros_like(curr3))
                # l = self.loss(curr, target)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                loss += l

            
        
        return loss.item()

    # def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
    #     """
    #         Update the parameters of the critic.

    #         let sum_of_path_lengths be the sum of the lengths of the paths sampled from
    #             Agent.sample_trajectories
    #         let num_paths be the number of paths sampled from Agent.sample_trajectories

    #         arguments:
    #             ob_no: shape: (sum_of_path_lengths, ob_dim)
    #             next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
    #             reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
    #                 the reward for each timestep
    #             terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
    #                 at that timestep of 0 if the episode did not end

    #         returns:
    #             training loss
    #     """
    #     # TODO: Implement the pseudocode below: do the following (
    #     # self.num_grad_steps_per_target_update * self.num_target_updates)
    #     # times:
    #     # every self.num_grad_steps_per_target_update steps (which includes the
    #     # first step), recompute the target values by
    #     #     a) calculating V(s') by querying the critic with next_ob_no
    #     #     b) and computing the target values as r(s, a) + gamma * V(s')
    #     # every time, update this critic using the observations and targets
    #     #
    #     # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
    #     #       to 0) when a terminal state is reached
    #     # HINT: make sure to squeeze the output of the critic_network to ensure
    #     #       that its dimensions match the reward

    #     ob_no = ptu.from_numpy(ob_no)
    #     next_ob_no = ptu.from_numpy(next_ob_no)
    #     ac_na = ptu.from_numpy(ac_na)
    #     reward_n = ptu.from_numpy(reward_n)
    #     terminal_n = ptu.from_numpy(terminal_n)


    #     loss = 0
    #     for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
    #         if i % self.num_grad_steps_per_target_update == 0:
    #             v = self(next_ob_no)
    #             target = (reward_n + self.gamma * v * (1-terminal_n)).detach()
                
    #         curr = self(ob_no)
    #         l = self.loss(curr, target)
    #         self.optimizer.zero_grad()
    #         l.backward()
    #         self.optimizer.step()
    #         loss += l
        
    #     return loss.item()
