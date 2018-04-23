from gym_orbit.envs import ActionLibrary as al
from gym_orbit.envs import StateLibrary as sl
from gym_orbit.envs import orbitalMotion as om
import numpy as np
<<<<<<< Updated upstream
from scipy import linalg as sci
=======
import scipy as sci
from scipy.linalg import expm
>>>>>>> Stashed changes
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('..')
from matplotlib import pyplot as plt
import orbitalMotion as om

def orbit_plot(rv_state):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(rv_state[0,:], rv_state[1,:], rv_state[2,:])
    ax.scatter([0], [0], [0], color="r", s=100)
    return fig

def state_plot(rv_state):
    fig, axarr = plt.subplots(6, sharex=True)
    yArr = ['$x$ (m)', '$y$ (m)', '$z$ (m)', '$\dot{x}$ (m/s)', '$\dot{y}$ (m/s)', '$\dot{z}$ (m/s)']
    for plotInd in range(0, 6):
        legendAppend = str(plotInd)
        if plotInd < 6:
            axarr[plotInd].plot(rv_state[plotInd, :])
        axarr[plotInd].set_ylabel(yArr[plotInd])
        axarr[plotInd].grid()
    axarr[-1].set_xlabel('Simulation Timestep')
    return fig

def states_plot(true_state, ref_state, est_state):
    fig, axarr = plt.subplots(6, sharex=True)
    yArr = ['$x$ (m)', '$y$ (m)', '$z$ (m)', '$\dot{x}$ (m/s)', '$\dot{y}$ (m/s)', '$\dot{z}$ (m/s)']
    for plotInd in range(0, 6):
        legendAppend = str(plotInd)
        if plotInd < 6:
            axarr[plotInd].plot(est_state[plotInd, :] - true_state[plotInd, :],label='estimator error')
            axarr[plotInd].plot(ref_state[plotInd, :] - true_state[plotInd, :],label='ctrl error')
            axarr[plotInd].plot(est_state[plotInd, :] - ref_state[plotInd, :], label='sc error')
        axarr[plotInd].set_ylabel(yArr[plotInd])
        axarr[plotInd].grid()
    plt.legend()
    axarr[-1].set_xlabel('Simulation Timestep')


def set_moi_ic():
    est_state = sl.observed_state()
    true_state = sl.rv_state()
    ref_state = sl.rv_state()

    mode_options = al.mode_options()
    mode_options.dt = 1.0
    mode_options.mode_length = 10.*60.0
    mode_options.mu = om.MU_MARS
    mode_options.j2 = 0#om.J2_MARS
    mode_options.rp = om.REQ_MARS
    mode_options.error_stm = sci.expm(mode_options.dt*(-0.01*np.identity(6)))

    desiredElements = om.ClassicElements()
    desiredElements.a = 1000
    desiredElements.e = 0.01
    desiredElements.Omega = 1.
    desiredElements.omega = 1.
    desiredElements.i = 1.
    desiredElements.f = 1.
    mode_options.goal_orbel = desiredElements
    mode_options.burn_number = 1
    mode_options.insertion_mode = True

    true_orbel = om.ClassicElements()
    true_orbel.a = 100000.0
    true_orbel.e = 0.8
    true_orbel.i = 0.0
    true_orbel.omega = 0.0
    true_orbel.Omega = 0.0
    true_orbel.f = -2.

    ref_orbel = om.ClassicElements()
    ref_orbel.a = 100000.0
    ref_orbel.e = 0.8
    ref_orbel.i = 0.0
    ref_orbel.omega = 0.0
    ref_orbel.Omega = 0.0
    ref_orbel.f = -2.

    est_orbel = om.ClassicElements()
    est_orbel.a = 100010.0
    est_orbel.e = 0.8
    est_orbel.i = 0.0
    est_orbel.omega = 0.0
    est_orbel.Omega = 0.0
    est_orbel.f = -2.

    ref_state.state_vec[0:3], ref_state.state_vec[3:] = om.elem2rv(om.MU_MARS, ref_orbel)
    true_state.state_vec[0:3], true_state.state_vec[3:] = om.elem2rv(om.MU_MARS, true_orbel)
    est_state.state_vec[0:3], est_state.state_vec[3:] = om.elem2rv(om.MU_MARS, est_orbel)
    return ref_state, est_state, true_state, mode_options

def set_default_ic():
    est_state = sl.observed_state()
    true_state = sl.rv_state()
    ref_state = sl.rv_state()

    mode_options = al.mode_options()
    mode_options.dt = 5.0
    mode_options.mode_length = 5.*60.0
    mode_options.mu = om.MU_MARS
    mode_options.j2 = 1.*om.J2_MARS
    mode_options.rp = om.REQ_MARS
<<<<<<< Updated upstream
    mode_options.error_stm = sci.linalg.expm(mode_options.dt*(-0.01*np.identity(6)))
=======
    mode_options.error_stm = expm(mode_options.dt*(-0.1*np.identity(6)))
    mode_options.obs_limit = 0.05
    mode_options.reward_multiplier = 100.0
>>>>>>> Stashed changes

    true_orbel = om.ClassicElements()
    true_orbel.a = 7100.0
    true_orbel.e = 0.01
    true_orbel.i = 0.0
    true_orbel.omega = 0.0
    true_orbel.Omega = 0.0
    true_orbel.f = 0.00

    ref_orbel = om.ClassicElements()
    ref_orbel.a = 7105.0
    ref_orbel.e = 0.01
    ref_orbel.i = 0.0
    ref_orbel.omega = 0.0
    ref_orbel.Omega = 0.0
    ref_orbel.f = 0.01

    est_orbel = om.ClassicElements()
    est_orbel.a = 7110.0
    est_orbel.e = 0.01
    est_orbel.i = 0.0
    est_orbel.omega = 0.0
    est_orbel.Omega = 0.0
    est_orbel.f = 0.000000

    ref_state.state_vec[0:3], ref_state.state_vec[3:] = om.elem2rv(om.MU_MARS, ref_orbel)
    true_state.state_vec[0:3], true_state.state_vec[3:] = om.elem2rv(om.MU_MARS, true_orbel)
    est_state.state_vec[0:3], est_state.state_vec[3:] = om.elem2rv(om.MU_MARS, est_orbel)
    return ref_state, est_state, true_state, mode_options

def test_propagators():

    ref_state, est_state, true_state, mode_options = set_default_ic()

    n_steps = 10000
    ref_hist = np.zeros([6,n_steps])
    ref_hist[:,0] = ref_state.state_vec
    est_hist = np.zeros([6,n_steps])
    est_hist[:,0] = est_state.state_vec
    true_hist = np.zeros([6,n_steps])
    true_hist[:,0] = true_state.state_vec

    for ind in range(1,n_steps):
        ref_state = al.sc_propagate(ref_state, mode_options)
        ref_hist[:,ind] = ref_state.state_vec

        true_state = al.truth_propagate(true_state, mode_options)
        true_hist[:, ind] = true_state.state_vec

        est_state = al.est_propagate(est_state, mode_options)
        est_hist[:, ind] = est_state.state_vec


    ref_fig = orbit_plot(ref_hist)
    est_fig = orbit_plot(est_hist)
    true_fig = orbit_plot(true_hist)

    plt.show()

def test_DV():

    # ref_state, est_state, true_state, mode_options = set_default_ic()
    ref_state, est_state, true_state, mode_options = set_moi_ic()

    n_steps = 100000
    ref_hist = np.zeros([6,n_steps])
    ref_hist[:,0] = ref_state.state_vec
    est_hist = np.zeros([6,n_steps])
    est_hist[:,0] = est_state.state_vec
    true_hist = np.zeros([6,n_steps])
    true_hist[:,0] = true_state.state_vec

    for ind in range(1,n_steps):
        ref_state = al.sc_htransfer_propagate(ref_state, mode_options)
        ref_hist[:,ind] = ref_state.state_vec

        true_state = al.truth_propagate(true_state, mode_options)
        true_hist[:, ind] = true_state.state_vec

        est_state = al.est_propagate(est_state, mode_options)
        est_hist[:, ind] = est_state.state_vec

        if ind == int(n_steps/2):
            true_state, DVtruth = al.resultOrbit(true_state, mode_options.goal_orbel)
            true_hist[:, ind] = true_state.state_vec

            est_state, DVest = al.resultOrbit(est_state, mode_options.goal_orbel)
            est_hist[:, ind] = est_state.state_vec

    ref_fig = orbit_plot(ref_hist)
    est_fig = orbit_plot(est_hist)
    true_fig = orbit_plot(true_hist)

    plt.show()

def test_obsMode():
    ref_state, est_state, true_state, mode_options = set_default_ic()

    n_steps = 1

    ref_hist = np.zeros([6, n_steps])
    true_hist = np.zeros([6, n_steps])
    est_hist = np.zeros([6, n_steps])

    for ind in range(0, n_steps):
        est_state, ref_state, true_state = al.observationMode(est_state, ref_state, true_state, mode_options)

        ref_hist[:, ind] = ref_state.state_vec
        est_hist[:, ind] = est_state.state_vec
        true_hist[:, ind] = true_state.state_vec

    states_fig = states_plot(true_hist, ref_hist, est_hist)
    plt.show()


def test_ctrlMode():
    ref_state, est_state, true_state, mode_options = set_default_ic()


    n_steps = 10

    ref_hist = np.zeros([6, n_steps])
    true_hist = np.zeros([6,n_steps])
    est_hist = np.zeros([6,n_steps])
    ctrl_hist = [0]

    for ind in range(0,n_steps):
        est_state, ref_state, true_state, control_use = al.controlMode(est_state, ref_state, true_state, mode_options)

        ref_hist[:,ind] = ref_state.state_vec
        est_hist[:, ind] = est_state.state_vec
        true_hist[:, ind] = true_state.state_vec
        ctrl_hist.append(control_use)


    states_fig = states_plot(true_hist, ref_hist, est_hist)
    ref_orb = orbit_plot(ref_hist)
    ref_state = state_plot(ref_hist)
    true_orb = orbit_plot(true_hist)
    true_state = state_plot(true_hist)
    est_orb = orbit_plot(est_hist)
    est_state = state_plot(est_hist)

    plt.figure()
    plt.plot(ctrl_hist)
    plt.title('Control use')
    plt.show()

def test_scienceMode():
    ref_state, est_state, true_state, mode_options = set_default_ic()


    n_steps = 10

    ref_hist = np.zeros([6, n_steps])
    true_hist = np.zeros([6,n_steps])
    est_hist = np.zeros([6,n_steps])
    reward_hist = []

    for ind in range(0,n_steps):
        est_state, ref_state, true_state, reward = al.scienceMode(est_state, ref_state, true_state, mode_options)

        ref_hist[:,ind] = ref_state.state_vec
        est_hist[:, ind] = est_state.state_vec
        true_hist[:, ind] = true_state.state_vec
        reward_hist.append(reward)


    states_fig = states_plot(true_hist, ref_hist, est_hist)
    ref_orb = orbit_plot(ref_hist)
    ref_state = state_plot(ref_hist)
    true_orb = orbit_plot(true_hist)
    true_state = state_plot(true_hist)
    est_orb = orbit_plot(est_hist)
    est_state = state_plot(est_hist)

    plt.figure()
    plt.plot(reward_hist)
    plt.title('Optional reward generated')
    plt.show()


if __name__ == "__main__":
    #test_propagators()
    #test_obsMode()
<<<<<<< Updated upstream
    # test_ctrlMode()
    test_DV()
=======
    #test_ctrlMode()
    test_scienceMode()
>>>>>>> Stashed changes

