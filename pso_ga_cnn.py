import numpy as np
import torch
import torch.nn as nn
import copy
import ptan
import multiprocessing as mp
import time
import collections
import gym


MAX_SEED = 2**32 - 1

def dim_weights(model):
    return sum([p.numel() for p in model.parameters()])


def make_env(game):
    return ptan.common.wrappers.wrap_dqn(gym.make(game))


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


class Individual:
    def __init__(self):


# out_item = (reward_max_p, speed_p)
OutputItem = collections.namedtuple('OutputItem', field_names=['top_children_p', 'frames','position'])


class Particle:
    def __init__(self, population=10,device='cpu', game="PongNoFrameskip-v4"):
        self.population = population
        self.mutation_step = 0.005
        # self.device = None
        self.game = game
        self.device = device
        # self.position = positon
        self.g_best = None
        self.g_best_value = None
        self.l_best_seed = None
        self.l_best = None
        self.l_best_value = None
        self.env = make_env(self.game)
        self.parent_net = self.init_uniform_parent()


    def init_uniform_parent(self):
        parent_net = Net(self.env.observation_space.shape, self.env.action_space.n)
        for p in parent_net.parameters():
            re_distribution = torch.tensor(np.random.uniform(low=-2, high=2, size=p.data.size()).astype(np.float32))
            p.data += re_distribution
        return parent_net


    def evaluate(self, net):
        frames = 0
        env_e = make_env(self.game)
        reward = 0.0
        while True:
            obs_v = torch.FloatTensor([np.array(obs, copy=False)]).to(self.device)
            act_prob = net(obs_v).to(self.device)
            acts = act_prob.max(dim=1)[1]
            obs, r, done, _ = env_e.step(acts.data.cpu().numpy()[0])
            reward += r
            frames += 4
            if done:
                break
        return reward, frames

    def update_g_best(self,g_best_seeds):
        self.g_best = self.build_net(g_best_seeds)

    # def mutate_net(self,seed):
    #     # new_net_m = Net(self.env_e.observation_space.shape, self.env_e.action_space.n).to(self.device)
    #     # new_net_m.load_state_dict(self.position)
    #     new_net_m = copy.deepcopy(self.parent_net)
    #     # seed = input_m
    #
    #     for p in new_net_m.parameters():
    #         np.random.seed(seed)
    #         noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32)).to(self.device)
    #         p.data += self.mutation_step * noise_t
    #
    #     reward, frames = self.evaluate(new_net_m)
    #     result = (seed, reward, frames)
    #     return result

    def work_func(self,seed):
        child_net = self.mutate_net(self.parent_net,seed)
        reward, frames = self.evaluate(child_net)
        result = (seed, reward, frames)
        return result

    def mutate_net(self, net, seed, copy_net=True):
        new_net = copy.deepcopy(net) if copy_net else net
        # np.random.seed(seed)
        for p in new_net.parameters():
            np.random.seed(seed)
            noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
            p.data += self.mutation_step * noise_t
        return new_net


    def build_g_best(self, seeds):
        torch.manual_seed(seeds[0])
        # self.mutate_net(seed,copy_net=False)
        net = Net(self.env.observation_space.shape, self.env.action_space.shape)
        for seed in seeds[1:]:
            net = self.mutate_net(net, seed, copy_net=False)
        return net

    # def build_net(self,seeds):
    #     new_net = copy.deepcopy(self.parent_net)
    #     for p in new_net.parameters():
    #         np.random.seed(seed)
    #         noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
    #         p.data += self.mutation_step * noise_t
    #
    #     return new_net

    # update particle's position
    def update_parent_position(self,g_best_seeds):
        # l_best = self.build_net(self.l_best_seed)
        g_best = self.build_g_best(g_best_seeds)
        for p,l,g in self.parent_net.parameters(),self.l_best.parameters(),g_best.parameters():
            r_g = np.random.uniform(low=0, high=1, size=p.data.size()).astype(np.float32)
            r_p = np.random.uniform(low=0, high=1, size=p.data.size()).astype(np.float32)
            v = self.chi * (self.phi_p * r_p * (l.data-p.data) \
                                 + self.phi_g * r_g * (g.data - p.data))
            p.data += v

    # just evolve 1 generation to find the best child
    def evolve_particle(self):
        # seeds = []
        # for _ in range(self.population):
        #     seed = np.random.randint(MAX_SEED)
        #     seeds.append(seed)
        input_m = [(np.random.randint(MAX_SEED),) for _ in range(self.population)]
        pool = mp.Pool(self.population)
        # (seed, reward, frames)
        result = pool.map(self.mutate_net,input_m)
        pool.close()
        pool.join()

        result.sort(key=lambda p: p[1], reverse=True)
        all_frames = sum([pair[2] for pair in result])
        if self.l_best_value < result[0][1]:
            self.l_best_seed = result[0][0]
            self.l_best_value = result[0][1]
            self.l_best = self.mutate_net(self.l_best_seed)

        best_seeds = self.parent_seeds.append(self.l_best_seed)

        return best_seeds, self.l_best_value, all_frames # OutputItem(top_children_p=top_children_p, frames=frames, postion=position)


# def init_particle(device,population,game):
#     particle = Particle(device,population,game)
#     return particle.evolve()

def call_particle(particle):
    return particle.evolve_particle()



class ParticleSwarm:
    def __init__(self, frames_limit=1000, size=10, game="PongNoFrameskip-v4",population=20, chi=0.72984, phi_p=2.05, phi_g=2.05):
        self.size = size
        self.chi = chi
        self.phi_p = phi_p
        self.phi_g = phi_g

        self.best_score = None
        self.best_position = None

        self.frames_limit = frames_limit
        self.population = population
        self.game = game
        self.result = None
        self.devices = []
        self.p_input = []

    # def create_swarm(self, swarm_number):
    #     init_particles = []
    #     # swarm = lib.ParticleSwarm(evaluate, dim=dim_weights(shape), size=30)
    #     for s in range(swarm_number):
    #         init_particle = Net(self.env.observation_space.shape, self.env.action_space.n)
    #         for p in init_particles.parameters():
    #             re_distribution = torch.tensor(np.random.uniform(low=-1, high=1, size=p.data.size()).astype(np.float32))
    #             p.data += re_distribution
    #         V[s] = np.random.uniform(low=-2, high=2, size=dim_weights(init_particle)).astype(np.float32)
    #
    #         # print("model.parameter.numbers:{}".format(sum([p.numel() for p in share_parent.parameters()])))
    #         P.append(copy.deepcopy(init_particle))
    #         if G is None:
    #             G = copy.deepcopy(init_particle)
    #         else:
    #             if evaluate(G) < evaluate(init_particle):
    #                 G = copy.deepcopy(init_particle)
    #         init_particles.append(init_particle)
    #
    #     return init_particles


    # create particles with different device
    def init_swarm(self):
        gpu_number = torch.cuda.device_count()
        # seed = np.random.randint(MAX_SEED)

        if gpu_number >= 1:
            for i in range(gpu_number):
                self.devices.append("cuda:{0}".format(i))

        for u in range(self.size):
            # seed = np.random.randint(MAX_SEED)
            if gpu_number == 0:
                device = "cpu"
            else:
                device_id = u % gpu_number
                device = self.devices[device_id]
            # particle_position = Net(self.env.observation_space.shape, self.env.action_space.n)
            # # np.random.seed(seed)
            # for p in particle_position.parameters():
            #     np.random.seed(seed)
            #     re_distribution = torch.tensor(
            #         np.random.uniform(low=-1, high=1, size=p.data.size()).astype(np.float32))
            #     p.data += re_distribution
            # init_position = particle_position.state_dict()
            p = Particle(device=device, population=self.population, game=self.game)
            self.p_input.append(p)

    # find the new best global among all particle
    def evolve_swarm(self):
        frames = 0
        # frames_per_g = 0
        devices = []
        gpu_number = torch.cuda.device_count()
        # elite = None
        if gpu_number >= 1:
            for i in range(gpu_number):
                devices.append("cuda:{0}".format(i))

        # t_start = time.time()

        # init particle create device
        self.init_swarm()
        # particle = Particle(population=10)

        while frames < self.frames_limit:
            pool = mp.Pool(self.size)

            # result = (l_best, reward, frames)
            self.result = pool.map(call_particle, self.p_input)
            pool.close()
            pool.join()
            # result  (l_best, reward, frames)
            self.result.sort(key=lambda p: p[1], reverse=True)
            frames = sum([pair[2] for pair in self.result])

            if self.result[0][1] > self.best_score:
                self.best_position = self.result[0][0]
                self.best_score = self.result[0][1]
                for particle in self.p_input:
                    # particle.update_g_best(self.best_position)
                    particle.update_parent_position(self.best_position)

            # max_value = max(key=lambda p: p[1])

if __name__ == '__main__':
    swarm = ParticleSwarm()
    swarm.evolve_swarm()
