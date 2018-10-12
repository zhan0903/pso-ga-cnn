import numpy as np
import torch
import torch.nn as nn
import copy
import ptan
# import multiprocessing as mp
import torch.multiprocessing as mp
import time
import collections
import gym.spaces
import gym
import sys
import json
import logging
from tensorboardX import SummaryWriter
import pickle


MAX_SEED = 2**32 - 1
mutation_step = 0.01


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


def evaluate(net, device, env_e):
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


def mutate_net(net, seed, device, loc=0, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    # np.random.seed(seed)
    # print("in mutate_net,Before, parent_net:{}".format(new_net.state_dict()['fc.2.bias']))
    np.random.seed(seed)
    if seed:
        for p in new_net.parameters():
            # np.random.seed(seed)
            noise_t = torch.tensor(np.random.normal(loc=loc, size=p.data.size()).astype(np.float32)).to(device)
            p.data += mutation_step * noise_t
    return new_net


def build_net(env, seeds, device):
    torch.manual_seed(seeds[0])
    # loc = seeds[0][1]
    net = Net(env.observation_space.shape, env.action_space.n).to(device)
    for idx, item in enumerate(seeds[1:]):
        if idx == 0:
            # print("item in build_net:{}".format(item))
            loc = item[0]
            seed = item[1]
        else:
            seed = item
            loc = 0
        # print("seed in build net:{}".format(seed))
        net = mutate_net(net, seed, device, loc=loc, copy_net=False)
    return net


def work_func(input_w):
    # work_id = mp.current_process()
    seed_w = input_w[0]
    game = input_w[1]
    device = input_w[2]
    # print("device in work_func:{}".format(device))
    # noise_step = input_w[3]
    # with open(r"my_trainer_objects.pkl", "rb") as input_file:
    #     parent_net = pickle.load(input_file)

    if device != "cpu":
        device_w_id = int(device[-1])
        torch.cuda.set_device(device_w_id)

    # print("in work_func, device:{},id:{}".format(device, work_id))

    env_w = make_env(game)
    parent_net_w = build_net(env_w, seed_w, device)
    # seed = np.random.randint(MAX_SEED)

    # torch.manual_seed(seed)
    # parent_net_w = Net(env_w.observation_space.shape, env_w.action_space.n).to(device)
    # parent_net_w.load_state_dict(parent_net)
    # print("in work_func,parent_net:{}".format(parent_net_w.state_dict()['fc.2.bias']))
    # print("in work_func,seed_w:{}".format(seed_w))

    # child_net = mutate_net(parent_net_w.to(device), seed_w, device, copy_net=False)
    # reward, frames = evaluate(child_net, device, env_w)
    reward, frames = evaluate(parent_net_w, device, env_w)

    result = (seed_w, reward, frames)
    # print("in work_func,reward:{}".format(reward))
    return result


class Particle:
    def __init__(self, logger, parents=[], g_best=None, parents_size=20, l_best_value=None, l_best=None, parent_net=None, velocity=None,
                 population=10, devices='cpu', loc=None, chi=0.72984, phi_p=2.05, phi_g=2.05, game="PongNoFrameskip-v4"):
        self.population = population
        self.chi = chi
        self.seeds = []
        self.phi_p = phi_p
        self.phi_g = phi_g
        # self.mutation_step = 0.005
        self.game = game
        self.devices = devices
        self.parents_size = parents_size
        self.g_best = g_best
        # self.g_best_value = g_best_value
        self.l_best_seed = None
        self.l_best = copy.deepcopy(l_best)
        self.l_best_value = l_best_value
        self.parent_net = copy.deepcopy(parent_net)
        self.env = make_env(self.game)
        self.logger = logger
        self.parents = parents
        self.elite = None
        self.velocity = copy.deepcopy(velocity)
        self.max_process = mp.cpu_count()  # mp.cpu_count()
        self.loc = loc

        # self.init_uniform_parent()

    def update_g_best(self, g_best_net):
        self.g_best = copy.deepcopy(g_best_net)

    # def build_g_best(self, seeds):
    #     torch.manual_seed(seeds[0])
    #     # self.mutate_net(seed,copy_net=False)
    #     net = Net(self.env.observation_space.shape, self.env.action_space.shape)#.to(self.device)
    #     for seed in seeds[1:]:
    #         net = mutate_net(net, seed, device="cpu", copy_net=False)
    #     return net

    def update_parents(self, parents):
        self.parents = parents

    # update particle's position
    def update_position(self):
        if self.g_best is None:
            self.g_best = self.parent_net
        for p, l, g, v in zip(self.parent_net.parameters(), self.l_best.parameters(),
                              self.g_best.parameters(), self.velocity.parameters()):
            # self.logger.debug("update_parent_position,p:{0},l:{1},g:{2},v:{3}".format(p.data, l.data, g.data, v.data))
            r_g = np.random.uniform(low=0, high=1, size=p.data.size()).astype(np.float32)
            r_p = np.random.uniform(low=0, high=1, size=p.data.size()).astype(np.float32)
            v = v*1 + self.chi * (self.phi_p * r_p * (l.data-p.data) + self.phi_g * r_g * (g.data - p.data))
            p.data += v

        # reward, frames = evaluate(self.parent_net, self.devices[0], self.env)
        # # if reward > self.l_best_value:
        # #     self.l_best_value = reward
        # #     self.l_best = copy.deepcopy(self.parent_net)
    def clone(self):
        self.logger.debug("best_seed in clone:{}".format(self.l_best_seed))
        for i in range(200):
            parent = np.random.randint(0, self.population)
            # parent = 0
            # self.logger.debug("best_seed in clone:{}".format(self.l_best_seed))
            self.seeds[parent] = copy.copy(self.l_best_seed)
            # self.logger.debug("self.seeds[parent] in clone:{}".format(self.seeds[i]))
        self.logger.debug("in clone,self.seeds len:{}".format(len(self.seeds)))

        # just evolve 1 generation to find the best child
    def evolve_particle(self):
        # input_m = []
        # input_seed = None
        # self.logger.debug("in evolve_particle, parent_net in particle:{}".
        #                   format(self.parent_net.state_dict()['fc.2.bias']))
        gpu_number = torch.cuda.device_count()
        # self.logger.debug("in evolve_particle, self.seeds:{}".format(self.seeds))
        # noise_step = None
        time_start = time.time()
        # init = False
        # while True:
        input_m = []
        self.logger.debug("in evolve_particle, self.seeds[:10]:{}".format(self.seeds[:10]))
        self.logger.debug("in evolve_particle, len of self.seeds:{}".format(len(self.seeds)))
        self.logger.debug("in evolve_particle, self.parents:{}".format(self.parents))

        for u in range(self.population):
            # noise_step = np.random.normal(scale=0.8)
            if gpu_number == 0:
                device = "cpu"
            else:
                device_id = u % gpu_number
                device = self.devices[device_id]
            # if u == self.population:
            #     input_m.append((None, self.game, device))
            # else:
            seed = np.random.randint(MAX_SEED)
            # if self.parents:
            #     seed2 = np.random.randint(MAX_SEED)
            #     input_individual = [seed, self.loc, seed2]
            #
            if not self.seeds or len(self.seeds) < self.population:
                seed2 = np.random.randint(MAX_SEED)
                self.seeds.append([seed, (self.loc, seed2)])
            else:
                # self.logger.debug("in evolve_particle, seed:{0},self.seeds[u]:{1}".format(seed, self.seeds[u]))
                self.seeds[u].append(seed)
            parent = np.random.randint(0, self.parents_size)
            if self.parents:
                input_seed = copy.copy(self.parents[parent])
                seed = np.random.randint(MAX_SEED)
                input_seed.append(seed)
            else:
                input_seed = self.seeds[u]

            # self.logger.debug("in evolve_paricle,u:{0},self.seeds[u]:{1}".format(u, self.seeds[u]))
            input_m.append((input_seed, self.game, device))
            # evaluate parent net
            # input_m.append((None, self.game, self.devices[0]))
            # with open(r"my_trainer_objects.pkl", "wb") as output_file:
            #     pickle.dump(self.parent_net.state_dict(), output_file, True)

        # max_process = max_cpu cores
        pool = mp.Pool(self.max_process)
        # (seed, reward, frames) map->map_aync
        result = pool.map(work_func, input_m)
        pool.close()
        pool.join()

        assert len(result) == self.population
        # if self.elite is not None:
        #     result.append(self.elite)
        # result.sort(key=lambda p: p[1], reverse=True)

        # if init:
        #     self.parents = [(item[0], item[1]) for item in result[:20]]
        #     self.logger.debug("self.parents:{}".format(self.parents))
        #     init = True

        # all_frames = sum([pair[2] for pair in result])
        # self.logger.info("current best score:{0},l_best_value:{1}".format(result[0][1],self.l_best_value))
        # self.logger.info("time cost:{}".format((time.time()-time_start)//60))
        # if self.l_best_value < result[0][1]:
            #     # init = True
            #
            #     # self.parents = copy.copy([(item[0], item[1]) for item in result[:20]])
            #     # self.logger.debug("self.parents:{}".format(self.parents))
            #     self.logger.debug("self.l_best_value:{}".format(self.l_best_value))
            #     self.elite = (result[0][0], result[0][1], 0)
            #     # self.l_best_seed = result[0][0]
            #     self.l_best_value = result[0][1]
                # self.clone()
                # self.logger.debug("self.seeds len:{0},self.seeds:{1}".format(len(self.seeds), self.seeds))
            # if init:
            #     self.parents = [(item[0], item[1]) for item in result[:20]]
            #     self.logger.debug("self.parents:{}".format(self.parents))

            # self.parents = []
            # for i in range(10):
            #     self.parents.append(result[i][0])
            #self.l_best = mutate_net(net=self.parent_net, device="cpu", seed=result[0][0])

        # best_seeds = self.parent_seeds.append(self.l_best_seed)
        # self.logger.info("in evolve_particle, best score in paritcle:{0}, seed:{1}".format(result[0][1], result[0][0]))
        return result


# def update_particle(particle):
#     return particle.update_parent_position()


class ParticleSwarm:
    def __init__(self, frames_limit=100000, parents_size=20, swarm_size=11, game="PongNoFrameskip-v4", population=20, logger=None):
        self.swarm_size = swarm_size
        self.best_score = None
        # self.best_seeds = None
        self.best_net = None
        self.logger = logger
        self.parents_size = parents_size

        self.frames_limit = frames_limit
        self.population = population
        self.game = game
        self.results = []
        self.devices = []
        self.p_input = []
        self.frames = 0
        self.env = make_env(game)
        self.elite = None

    def init_swarm(self):
        devices = []
        gpu_number = torch.cuda.device_count()
        # seed = np.random.randint(MAX_SEED)
        if gpu_number >= 1:
            for i in range(gpu_number):
                devices.append("cuda:{0}".format(i))
            # for _ in range(gpu_number):
            #     devices.append("cuda:{0}".format(1))
        else:
            devices = "cpu"
        # create normal parents_net
        loc = 0
        loc_limit = (self.swarm_size-1)//2
        # self.particles = [Particle(loc=i) for i in range(self.swarm_size)]

        for u in range(self.swarm_size):

            # # create normal parents_net
            # particle_parent_net = Net(self.env.observation_space.shape, self.env.action_space.n)  # .to(self.device)
            # for p in particle_parent_net.parameters():
            # #     re_distribution = torch.tensor(np.random.normal(loc=loc, size=p.data.size()).astype(np.float32))#.to(device)
            # #     p.data += re_distribution
            # if loc == loc_limit:
            #     loc = -loc_limit
            # loc = loc + 1
            # seed = np.random.randint(MAX_SEED)

            # # self.parent_net = parent_net
            # self.logger.debug("in init_swarm, create_normal_parent, particle_parent_net:{}".
            #                   format(particle_parent_net.state_dict()['fc.2.bias']))
            # # reward, frames = evaluate(particle_parent_net, "cpu", self.env)
            # # self.frames = self.frames+frames
            # if self.best_score is None:
            #     self.best_score = reward
            #     # self.best_net = copy.deepcopy(particle_parent_net)
            #
            # if reward > self.best_score:
            #     self.best_score = reward
            #     self.best_net = copy.deepcopy(particle_parent_net)
            # l_best_value = reward
            # l_best = particle_parent_net
            # velocity = Net(self.env.observation_space.shape, self.env.action_space.n)
            # self.logger.debug("in init_swarm,l_best_value:{}".format(l_best_value))

            p = Particle(logger=self.logger, loc=loc, parents_size=self.parents_size,  devices=devices,
                         population=self.population, game=self.game)
            self.p_input.append(p)
            if loc == loc_limit:
                loc = -loc_limit
            else:
                loc = loc + 1

    def update_particle(self, particle):
        particle.update_parent_position(self.best_net)

    # find the new best global among all particle
    def evolve_swarm(self):
        # particle = Particle(population=10)
        self.init_swarm()
        time_start = time.time()
        writer = SummaryWriter(comment="-pong-ga-multi-species")

        while self.frames < self.frames_limit:
            # evolve particle
            self.results = []
            # self.logger.debug("self.p_input:{}".format(self.p_input))
            for idx, particle in enumerate(self.p_input):
                # self.logger.debug("in evolve_swarm, particle idx:{0},particle parent net:{1}".
                #                   format(idx, particle.parent_net.state_dict()['fc.2.bias']))
                # result_particle/(seed,reward), all_frames
                # [(seed, reward, frames),]
                result = particle.evolve_particle()
                self.results.extend(result)
            self.logger.debug("in evolve_swarm, len self.results;{}".format(len(self.results)))
            self.logger.debug("in evolve_swarm, self.results:{}".format(self.results[:10]))
            frames = sum([result[2] for result in self.results])
            if self.elite is not None:
                self.results.append(self.elite)
            self.results.sort(key=lambda p: p[1], reverse=True)
            self.frames = self.frames+frames
            if self.best_score is None:
                self.best_score = self.results[0][1]

            if self.results[0][1] > self.best_score:
                # no need deep copy here
                self.best_net = self.results[0][0]
                self.best_score = self.results[0][1]
                self.elite = self.results[0]
                new_parents = [item[0] for item in self.results[:self.parents_size]]
                self.logger.debug("in evolve_swarm, new_parents:{}".format(new_parents))
                for particle in self.p_input:
                    particle.update_parents(new_parents)

            self.logger.info("in evolve_swarm, best core:{}".format(self.best_score))
            self.logger.info("in evolve_swarm, time cost:{}ms".format((time.time() - time_start)/60))

        self.logger.info("whole time cost:{}ms".format((time.time()-time_start)/60))


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
    parents_size = exp["parents_size"]

    if exp["frames_limit"][-1] == "B":
        frames_limit = 1000000000*int(exp["frames_limit"][:-1])
    elif exp["frames_limit"][-1] == "M":
        frames_limit = 1000000*int(exp["frames_limit"][:-1])
    else:
        frames_limit = int(exp["frames_limit"])

    logger.info("{}".format(str(json.dumps(exp, indent=4, sort_keys=True))))
    for game in games:
        swarm = ParticleSwarm(frames_limit=frames_limit, parents_size=parents_size, game=game, swarm_size=swarm_size,
                              population=population_per_worker, logger=logger)
        swarm.evolve_swarm()
        logger.info("game=%s,reward_max=%.2f" % (game, swarm.best_score))


if __name__ == '__main__':
    with open(sys.argv[-1], 'r') as f:
        exp = json.loads(f.read())
    main(**exp)
