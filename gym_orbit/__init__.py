import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='linear_orbit-v0',
    entry_point='gym_orbit.envs:LinearOrbitEnv',
)
register(
    id='stationkeep_orbit-v0',
    entry_point='gym_orbit.envs:pls_work_env',
)
register(
    id='mars_orbit_insertion-v0',
    entry_point='gym_orbit.envs:mars_orbit_insertion',
)
