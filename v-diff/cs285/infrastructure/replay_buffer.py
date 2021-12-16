from cs285.infrastructure.utils import *


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def add_rollouts(self, paths, noised=False):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, next_observations, terminals, concatenated_rews, concatenated_next_rews, unconcatenated_rews, unconcatenated_next_rewards = convert_listofrollouts(paths)

        if noised:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.concatenated_rews = concatenated_rews[-self.max_size:]
            self.concatenated_next_rews = concatenated_next_rews[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]
            self.concatenated_rews = np.concatenate(
                [self.concatenated_rews, concatenated_rews]
            )[-self.max_size:]
            self.concatenated_next_rews = np.concatenate(
                [self.concatenated_next_rews, concatenated_next_rews]
            )[-self.max_size:]

    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert self.obs.shape[0] == self.acs.shape[0] == self.concatenated_rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0]
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.concatenated_rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]

    # def sample_recent_data(self, batch_size=1, concat_rew=True):

    #     if concat_rew:
    #         return self.obs[-batch_size:], self.acs[-batch_size:], self.concatenated_rews[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:]
    #     else:
    #         num_recent_rollouts_to_return = 0
    #         num_datapoints_so_far = 0
    #         index = -1
    #         while num_datapoints_so_far < batch_size:
    #             recent_rollout = self.paths[index]
    #             index -=1
    #             num_recent_rollouts_to_return +=1
    #             num_datapoints_so_far += get_pathlength(recent_rollout)
    #         rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
    #         observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews = convert_listofrollouts(rollouts_to_return)
    #         return observations, actions, unconcatenated_rews, next_observations, terminals
    
    def sample_recent_data(self, batch_size=1, concat_rew=True):
        if concat_rew:
            observations2 = self.obs[-batch_size:]
            actions2 = self.acs[-batch_size:]
            concatenated_rews2 = self.concatenated_rews[-batch_size:]
            next_obs2 = self.next_obs[-batch_size:]
            terminals2 = self.terminals[-batch_size:]

            observations1 = self.obs[-2 * batch_size:-batch_size]
            actions1 = self.acs[-2 * batch_size:-batch_size]
            concatenated_rews1 = self.concatenated_rews[-2 * batch_size:-batch_size]
            next_obs1 = self.next_obs[-2 * batch_size:-batch_size]
            terminals1 = self.terminals[-2 * batch_size:-batch_size]
            concatenated_next_rews1 = self.concatenated_next_rews[-2 * batch_size:-batch_size]

            return observations1, actions1, concatenated_rews1, concatenated_next_rews1, next_obs1, terminals1, observations2, actions2, concatenated_rews2, next_obs2, terminals2
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -=1
                num_recent_rollouts_to_return +=1
                num_datapoints_so_far += get_pathlength(recent_rollout)

            rollouts_to_return2 = self.paths[-num_recent_rollouts_to_return:]
            rollouts_to_return1 = self.paths[-2 * num_recent_rollouts_to_return : -num_recent_rollouts_to_return]
            observations1, actions1, next_observations1, terminals1, concatenated_rews1,concatenated_next_rews1, unconcatenated_rews1, unconcatenated_next_rews1 = convert_listofrollouts(rollouts_to_return1)
            observations2, actions2, next_observations2, terminals2, concatenated_rews2, concatenated_next_rews2, unconcatenated_rews2, unconcatenated_next_rews2 = convert_listofrollouts(rollouts_to_return2)

            return observations1, actions1, unconcatenated_rews1, unconcatenated_next_rews1, next_observations1, terminals1, observations2, actions2, unconcatenated_rews2, next_observations2, terminals2
