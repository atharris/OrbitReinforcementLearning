import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='linear_orbit-v0',
    entry_point='src.envs:LinearOrbitEnv',
)
