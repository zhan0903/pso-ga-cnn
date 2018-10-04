from pso_ga_cnn import ParticleSwarm
import pytest


class Algorithm(ParticleSwarm):
    """
    Tries to get a randomly-generated list to match [.1, .2, .3, .2, .1]
    """

    def _result(self):
        pass


@pytest.mark.create_swarm
def test_create_swarm():
    algorithm = Algorithm(swarm_size=2)
    algorithm.create_swarm()
    assert len(algorithm.p_input) == 2


@pytest.mark.init_swarm
def test_init_swarm():
    algorithm = Algorithm(swarm_size=2)
    algorithm.evolve_swarm()


@pytest.mark.evolve_swarm
def test_evolve_swarm():
    algorithm = Algorithm(swarm_size=2)
    algorithm.evolve_swarm()




