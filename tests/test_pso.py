from pso_ga_cnn import ParticleSwarm
import pytest
import logging
import multiprocessing as mp
import torch


class Algorithm(ParticleSwarm):
    """
    Tries to get a randomly-generated list to match [.1, .2, .3, .2, .1]
    """

    def _result(self):
        pass


@pytest.mark.create_swarm
def test_create_swarm():
    algorithm = Algorithm(swarm_size=4)
    algorithm.create_swarm()
    assert len(algorithm.p_input) == 2


@pytest.mark.init_swarm
def test_init_swarm():
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler('./logger.out')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    mp.set_start_method('spawn')
    swarm = Algorithm(logger=logger)
    # swarm.create_swarm()
    swarm.init_swarm()
    assert len(swarm.p_input) == 11
    for particle in swarm.p_input:
        assert swarm.best_score is not None
        # assert swarm.best_net is not None
        assert particle is not None

        # swarm_best_net = swarm.best_net.state_dict()['fc.2.bias']
        # particle_best_net = particle.g_best.state_dict()['fc.2.bias']
        # print("particle_best_net:{}".format(particle_best_net))
        # print("swarm_best_net:{}".format(swarm_best_net))

        # assert torch.equal(swarm_best_net, particle_best_net)


@pytest.mark.evolve_swarm
def test_evolve_swarm():
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler('./logger.out')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    mp.set_start_method('spawn')
    swarm = Algorithm(logger=logger)
    # swarm.create_swarm()
    # swarm.init_swarm()
    swarm.evolve_swarm()
    assert len(swarm.p_input) == 11
    print("swarm.best_score:{}".format(swarm.best_score))
    print("swarm.best_")









