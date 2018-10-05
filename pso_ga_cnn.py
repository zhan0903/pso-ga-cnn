import numpy as np
import torch
import torch.nn as nn
import copy
import ptan
import multiprocessing as mp
# import torch.multiprocessing as mp
import time
import collections
import gym.spaces
import gym
import sys
import json
import logging


MAX_SEED = 2**32 - 1
mutation_step = 0.005


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


# out_item = (reward_max_p, speed_p)
OutputItem = collections.namedtuple('OutputItem', field_names=['top_children_p', 'frames','position'])


def work_func(self, seed):
    child_net = self.mutate_net(self.parent_net, seed)
    reward, frames = self.evaluate(child_net)
    result = (seed, reward, frames)
    return result


def evaluate(net, device,env_e):
    frames = 0
    # env_e = make_env(game)
    obs = env_e.reset()
    reward = 0.0
    while True:
        obs_v = torch.FloatTensor([np.array(obs, copy=False)]).to(device)
        act_prob = net(obs_v)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env_e.step(acts.data.cpu().numpy()[0])
        reward += r
        frames += 4
        if done:
            break
    return reward, frames


def mutate_net(net, seed, device, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    # np.random.seed(seed)
    print("in mutate_net,Before, parent_net:{}".format(new_net.state_dict()['fc.2.bias']))
    for p in new_net.parameters():
        np.random.seed(seed)
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32)).to(device)
        p.data += mutation_step * noise_t

    print("in mutate_net,After, parent_net:{}".format(new_net.state_dict()['fc.2.bias']))

    return new_net


def work_func(input_w):
    seed_w = input_w[0]
    parent_net = input_w[1]
    game = input_w[2]
    device = input_w[3]
    env_w = make_env(game)
    parent_net_w = Net(env_w.observation_space.shape, env_w.action_space.n)
    # print("in work_fun,device:{}".format(device))
    # print("in work_func, parent_net:{}".format(parent_net['fc.2.bias']))
    parent_net_w.load_state_dict(parent_net)
    child_net = mutate_net(parent_net_w.to(device), seed_w, device, copy_net=False)#.to(device)
    reward, frames = evaluate(child_net, device, env_w)#.to(device)
    result = (seed_w, reward, frames)
    return result


class Particle:
    def __init__(self, logger, population=10, device='cpu', game="PongNoFrameskip-v4"):
        self.population = population
        self.mutation_step = 0.005
        self.game = game
        self.device = device
        self.g_best = None
        self.g_best_value = None
        # self.l_best_seed = None
        self.l_best = None
        self.l_best_value = None
        self.parent_net = None
        self.env = make_env(self.game)
        self.logger = logger
        # self.init_uniform_parent()

    def create_uniform_parent(self):
        self.parent_net = Net(self.env.observation_space.shape, self.env.action_space.n)
        for p in self.parent_net.parameters():
            re_distribution = torch.tensor(np.random.uniform(low=-2, high=2, size=p.data.size()).astype(np.float32))
            p.data += re_distribution

        # self.parent_net = parent_net
        self.logger.debug("create_uniform_parent, parent_net:{}".format(self.parent_net.state_dict()['fc.2.bias']))
        reward, frames = evaluate(self.parent_net, self.env)
        self.l_best_value = reward
        self.l_best = self.parent_net
        return self.parent_net, reward, frames

    # def evaluate(self, net):
    #     frames = 0
    #     env_e = make_env(self.game)
    #     obs = env_e.reset()
    #     reward = 0.0
    #     while True:
    #         obs_v = torch.FloatTensor([np.array(obs, copy=False)]).to(self.device)
    #         act_prob = net(obs_v).to(self.device)
    #         acts = act_prob.max(dim=1)[1]
    #         obs, r, done, _ = env_e.step(acts.data.cpu().numpy()[0])
    #         reward += r
    #         frames += 4
    #         if done:
    #             break
    #     return reward, frames

    def update_g_best(self, g_best_net):
        self.g_best = g_best_net

    def return_parent_net(self):
        return self.parent_net

    # def work_func(self, seed):
    #     child_net = self.mutate_net(self.parent_net, seed)
    #     reward, frames = self.evaluate(child_net)
    #     result = (seed, reward, frames)
    #     return result

    # def mutate_net(self, net, seed, copy_net=True):
    #     new_net = copy.deepcopy(net) if copy_net else net
    #     # np.random.seed(seed)
    #     for p in new_net.parameters():
    #         np.random.seed(seed)
    #         noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32)).to(self.device)
    #         p.data += self.mutation_step * noise_t
    #     return new_net

    def build_g_best(self, seeds):
        torch.manual_seed(seeds[0])
        # self.mutate_net(seed,copy_net=False)
        net = Net(self.env.observation_space.shape, self.env.action_space.shape).to(self.device)
        for seed in seeds[1:]:
            net = self.mutate_net(net, seed, self.device, copy_net=False)
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
    def update_parent_position(self, g_best_seeds):
        # l_best = self.build_net(self.l_best_seed)
        g_best = self.build_g_best(g_best_seeds)
        for p, l, g in self.parent_net.parameters(), self.l_best.parameters(), g_best.parameters():
            r_g = np.random.uniform(low=0, high=1, size=p.data.size()).astype(np.float32)
            r_p = np.random.uniform(low=0, high=1, size=p.data.size()).astype(np.float32)
            v = self.chi * (self.phi_p * r_p * (l.data-p.data) + self.phi_g * r_g * (g.data - p.data))
            p.data += v

    # just evolve 1 generation to find the best child
    def evolve_particle(self):
        # mp.set_start_method('spawn')
        input_m = []
        self.logger.debug("in evolve_particle self.population:{}".format(self.population))
        self.logger.debug("in evolve_particle,self.parent_net['fc.2.bias']:{}".
                          format(self.parent_net.state_dict()['fc.2.bias']))
        net_test = self.return_parent_net()
        self.logger.debug("in evolve_particle,net_test['fc.2.bias']:{}".
                          format(net_test.state_dict()['fc.2.bias']))
        for _ in range(self.population):
            seed = np.random.randint(MAX_SEED)
            parent_net = self.parent_net.state_dict()
            # self.logger.debug("parent_net[0]['fc.2.bias']:{}".format(parent_net['fc.2.bias']))

            input_m.append((seed, parent_net, self.game, self.device))
        # input_m = [(np.random.randint(MAX_SEED),) for _ in range(self.population)]

        # self.logger.debug("parent_net[0]['fc.2.bias']:".format(input_m[0][1]['fc.2.bias']))
        pool = mp.Pool(self.population)
        # (seed, reward, frames)
        result = pool.map(work_func, input_m)
        pool.close()
        pool.join()

        result.sort(key=lambda p: p[1], reverse=True)
        all_frames = sum([pair[2] for pair in result])
        if self.l_best_value < result[0][1]:
            # self.l_best_seed = result[0][0]
            self.l_best_value = result[0][1]
            self.l_best = mutate_net(self.parent_net, result[0][0])

        # best_seeds = self.parent_seeds.append(self.l_best_seed)
        return self.l_best, self.l_best_value, all_frames


def update_particle(particle):
    return particle.update_parent_position()


class ParticleSwarm:
    def __init__(self, frames_limit=1000, swarm_size=2, game="PongNoFrameskip-v4", population=20, chi=0.72984, phi_p=2.05, phi_g=2.05,logger=None):
        self.swarm_size = swarm_size
        self.chi = chi
        self.phi_p = phi_p
        self.phi_g = phi_g

        self.best_score = None
        # self.best_seeds = None
        self.best_net = None
        self.logger = logger

        self.frames_limit = frames_limit
        self.population = population
        self.game = game
        self.results = []
        self.devices = []
        self.p_input = []
        self.frames = None

    # create particles with different device
    def create_swarm(self):
        gpu_number = torch.cuda.device_count()
        # seed = np.random.randint(MAX_SEED)

        if gpu_number >= 1:
            for i in range(gpu_number):
                self.devices.append("cuda:{0}".format(i))

        for u in range(self.swarm_size):
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
            p = Particle(logger=self.logger, device=device, population=self.population, game=self.game)
            self.p_input.append(p)

    def init_swarm(self):
        for particle in self.p_input:
            # parent_net, reward, frame
            result = particle.create_uniform_parent()
            self.results.append(result)

        self.results.sort(key=lambda p: p[1], reverse=True)
        self.frames = sum([pair[2] for pair in self.results])
        self.best_score = self.results[0][1]
        self.best_net = self.results[0][0]

        for particle in self.p_input:
            self.logger.debug("in init_swarm before:{}".format(particle.parent_net.state_dict()['fc.2.bias']))
            particle.update_g_best(self.best_net)
            self.logger.debug("in init_swarm after:{}".format(particle.parent_net.state_dict()['fc.2.bias']))

    # find the new best global among all particle
    def evolve_swarm(self):
        # init particle create device
        self.create_swarm()
        # particle = Particle(population=10)
        self.init_swarm()
        time_start = time.time()
        while self.frames < self.frames_limit:
            # input_t = [1, 2, 3, 4, 5, 6, 7, 8]
            # pool = mp.Pool(self.population)
            # # (seed, reward, frames)
            # result = pool.map(update_particle, input_t)
            # pool.close()
            # pool.join()

            # evolve particle
            self.results = []
            self.logger.debug("self.p_input:{}".format(self.p_input))
            for particle in self.p_input:
                self.logger.debug("in evolve_swarm, particle:{}".format(particle.parent_net.state_dict()['fc.2.bias']))
                self.logger.debug("in evolve_swarm, return particle:{}".format(particle.return_parent_net().state_dict()['fc.2.bias']))

                result = particle.evolve_particle()
                self.results.append(result)

            self.results.sort(key=lambda p: p[1], reverse=True)
            frames = sum([result[2] for result in self.results])
            self.frames = self.frames+frames

            if self.results[0][1] > self.best_score:
                self.best_net = self.result[0][0]
                self.best_score = self.result[0][1]
        self.logger.info("time cost:{}".format(time.time()-time_start))


def main(**exp):
    mp.set_start_method('spawn')
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler('./logger.out')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if exp["debug"]:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    swarm_size = exp["swarm_size"]
    population_per_worker = exp["population_per_worker"]
    games = exp["games"].split(',')
    logger.info("games:{}".format(games))

    if exp["frames_limit"][-1] == "B":
        frames_limit = 1000000000*int(exp["frames_limit"][:-1])
    elif exp["frames_limit"][-1] == "M":
        frames_limit = 1000000*int(exp["frames_limit"][:-1])
    else:
        frames_limit = int(exp["frames_limit"])

    logger.info("{}".format(str(json.dumps(exp, indent=4, sort_keys=True))))
    for game in games:
        swarm = ParticleSwarm(frames_limit=frames_limit, game=game, swarm_size=swarm_size,
                              population=population_per_worker, logger=logger)
        swarm.evolve_swarm()
        logger.info("game=%s,reward_max=%.2f" % (game, swarm.best_score))


if __name__ == '__main__':
    with open(sys.argv[-1], 'r') as f:
        exp = json.loads(f.read())
    main(**exp)
