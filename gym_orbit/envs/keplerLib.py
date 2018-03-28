import numpy as np
import math
import scipy as sci
import scipy.optimize as sciopt
import schaubLib as sl

def cot(angl):
    cotAngl = 1. / math.tan(angl)
    return cotAngl

class odeOptions:
    def __init__(self, mu):
        self.mu = mu

def orbitODE(y0,t,options):
    yd0 = np.zeros([6,])
    yd0[0:3] = y0[3:]
    #print yd0
    yd0[3:] = -options.mu / np.linalg.norm(y0[0:3])**3.0 * y0[0:3]
    #print yd0
    return yd0

class solverArgs:
    def __init__(self):
        self.tol = 1e-6
        self.maxiter = int(1000)

def au2meters(au):
    meters = 1.496e+11 * au
    return meters

def min2sec(min):
    sec = min*60
    return sec

def earthRadius():
    return 6378.137e3

def planetRadius(planet):
    if planet == "Earth":
        return 6378.1370e3
    elif planet == "Sun":
        return 1.327124280e20
    elif planet == "Mecury":
        return 2439.0e3
    elif planet == "Venus":
        return 6052e3
    elif planet == "Mars":
        return 3397.2e3
    elif planet == "Jupiter":
        return 71492e3
    elif planet == "Saturn":
        return 60268e3
    elif planet == "Uranus":
        return 22559e3
    elif planet == "Neptune":
        return 24764e3
    elif planet == "Moon":
        return 1740e3
    else:
        return -1

def planetYear(planet):
    if planet == "Earth":
        return 0.99997862
    elif planet == "Mercury":
        return 0.24084445
    elif planet == "Venus":
        return 0.61518257
    elif planet == "Mars":
        return 1.88071105
    elif planet == "Jupiter":
        return 11.856525
    elif planet == "Saturn":
        return 29.423519
    elif planet == "Uranus":
        return 83.747406
    elif planet == "Neptune":
        return 163.7232045
    elif planet == "Sun":
        return "What are you doing?"
    elif planet == "Moon":
        return 0.99997862
    else:
        return -1

def gravParam(planet):
    if planet == "Earth":
        return 3.986004415000000e14
    elif planet == "Mercury":
        return 2.2032e13
    elif planet == "Venus":
        return 3.257e14
    elif planet == "Mars":
        return 4.305e13
    elif planet == "Jupiter":
        return 1.268e17
    elif planet == "Saturn":
        return 3.794e16
    elif planet == "Uranus":
        return 5.794e15
    elif planet == "Neptune":
        return 6.809e15
    elif planet == "Sun":
        return 1.327124280e20
    elif planet == "Moon":
        return 4902.799e9
    else:
        return -1

def orbelConverter(inputType, outputType, elList, mu):
    outputEls = np.zeros(6,)
    if inputType == 'classical' and outputType == 'rv':
        outputEls[0:3], outputEls[3:] = orbel2rv(elList[0], elList[1], elList[2], elList[3], elList[4], elList[5], mu)

    elif inputType == 'rv' and outputType == 'classical':
        semi, ecc, incl, periArg, nodeArg, trueAnom = rv2orbel(elList[0:3], elList[3:], mu)
        outputEls[0] = semi
        outputEls[1] = ecc
        outputEls[2] = incl
        outputEls[3] = periArg
        outputEls[4] = nodeArg
        outputEls[5] = trueAnom

    return outputEls

def keplersEquation(eccAnom, meanAnom, ecc):
    err = meanAnom - (eccAnom - ecc*math.sin(eccAnom))
    return err

def keplersDerivative(eccAnom, meanAnom,ecc):
    derr = ( ecc*math.cos(eccAnom) - 1)
    return derr

def mean2ecc(meanAnom, ecc):
    eccAnom = sciopt.newton(keplersEquation,meanAnom,keplersDerivative,args=(meanAnom,ecc), maxiter=100)
    return eccAnom

def meanMotionComp(semi,mu):
    if semi > 0:
        meanMotion = math.sqrt(mu/(semi**(3.0)))
    else:
        meanMotion = math.sqrt(mu / (-semi ** (3.0)))
    return meanMotion

def meanMotion2semi(meanMotion, mu):
    semi = (mu / (meanMotion**2))**(1./3)
    return semi

def semi2period(semi,mu):
    period = 2*math.pi * math.sqrt((semi**3)/mu)
    return period

def orbitMomentumDir(orbitState):
    momentumVecDir = np.cross(orbitState[0:3],orbitState[3:])/np.linalg.norm(np.cross(orbitState[0:3],orbitState[3:]))
    return momentumVecDir

def meanAnomProp(initAnom,semi,mu,dt):
    meanMotion = meanMotionComp(semi,mu)
    currentAnom = initAnom + meanMotion*dt
    return currentAnom

def ecc2true(eccAnom, ecc):
    equalTo = math.sqrt((1+ecc)/(1-ecc))*math.tan(eccAnom/2)
    trueAnom = 2 * math.atan2(equalTo, 1)
    if np.sign(trueAnom) == -1:
        trueAnom = trueAnom + 2*math.pi
    return trueAnom

def true2ecc(trueAnom, ecc):
    trueTan = math.atan2(trueAnom,2)
    equalTo = math.sqrt((1-ecc)/(1+ecc)) * math.tan(trueAnom/2)
    eccAnom = 2*math.atan2(equalTo,1)

    if np.sign(eccAnom) == -1:
        eccAnom = eccAnom + 2*math.pi
    return eccAnom

def ecc2mean(eccAnom,ecc):
    meanAnom = eccAnom - ecc*math.sin(eccAnom)
    if np.sign(meanAnom) == -1:
        meanAnom = meanAnom + 2*math.pi

    return meanAnom

def keplersHyperEquation(hyperAnom, meanAnom, ecc):
    err = meanAnom + (hyperAnom - ecc*math.sin(hyperAnom))
    return err

def keplersHyperDerivative(hyperAnom, meanAnom,ecc):
    derr = ( - ecc*math.cos(hyperAnom) + 1)
    return derr

def mean2hyper(meanAnom, ecc):
    hyperAnom = sciopt.newton(keplersHyperEquation,meanAnom,keplersHyperDerivative,args=(meanAnom,ecc), maxiter=100)
    return hyperAnom

def hyper2mean(hyperAnom, ecc):
    meanAnom = ecc * math.sinh(hyperAnom) - hyperAnom
    return meanAnom

def hyper2true(hyperAnom, ecc):
    equalTo = math.sqrt((1.+ecc)/(ecc-1.))*math.tanh(hyperAnom/2.)
    trueAnom = 2. * math.atan2(equalTo, 1.)
    if np.sign(trueAnom) == -1:
        trueAnom = trueAnom + 2*math.pi
    return trueAnom

def mean2dt(meanAnom, initAnom, semi,mu):
    meanMotion = meanMotionComp(semi,mu)
    dt = (meanAnom - initAnom) / meanMotion
    return dt


def rv2orbel(posVec, velVec, mu):
    posNorm = np.linalg.norm(posVec)
    velNorm = np.linalg.norm(velVec)
    angVelVec = np.cross(posVec, velVec,axis=0)
    angVelNorm = np.linalg.norm(angVelVec)
    semi = (2/posNorm - (velNorm**2)/mu)**-1
    eccVec = np.cross(velVec, angVelVec,axis=0) / mu - posVec / posNorm
    ecc = np.linalg.norm(eccVec)

    ecc_unit = np.multiply(eccVec, 1/ecc)
    ang_unit = np.multiply(angVelVec, 1/angVelNorm)
    peri_unit = np.cross(ang_unit, ecc_unit,axis=0)
    attMat = (np.vstack([[ecc_unit],[peri_unit],[ang_unit]]))

    incl = math.acos(attMat[2, 2]) % (math.pi * 2.0)

    nodeArg = math.atan2(attMat[2,0], -attMat[2,1]) % (math.pi * 2.0)
    periArg = math.atan2(attMat[0,2],attMat[1,2]) % (math.pi * 2.0)

    sig_0 = np.inner(posVec, velVec)/math.sqrt(mu)
    if ecc < 1.0:
        eccAnom = math.atan2(sig_0/math.sqrt(semi), 1 - posNorm/semi)
        meanAnom = (ecc2mean(eccAnom, ecc)) % (math.pi * 2.0)
        trueAnom = ecc2true(eccAnom,ecc)
    else:
        hyperAnom = math.atan2(sig_0/math.sqrt(-semi), 1 - posNorm/semi)
        meanAnom = (hyper2mean(hyperAnom, ecc)) % (math.pi * 2.0)

    return semi, ecc, incl, nodeArg, periArg, meanAnom

def rot1(angle):
    rotMat = np.array([[1,0,0],
                       [0,math.cos(angle),math.sin(angle)],
                      [0, -math.sin(angle),math.cos(angle)]])
    return rotMat

def rot2(angle):
    rotMat = np.array([[math.cos(angle), 0, -math.sin(angle)],
                       [0, 1, 0],
                      [math.sin(angle), 0, math.cos(angle)]])
    return rotMat

def rot3(angle):
    rotMat = np.array([[math.cos(angle), math.sin(angle), 0],
                       [-math.sin(angle), math.cos(angle), 0],
                      [0, 0 , 1]])
    return rotMat

def orbel2rv(semi, ecc, incl, periArg, nodeArg, meanAnom, mu):
    if ecc < 1.0 and ecc >= 0.0:
        meanMotion = meanMotionComp(semi,mu)
        eccAnom = mean2ecc(meanAnom, ecc)
        trueAnom = ecc2true(eccAnom, ecc)
        latus = semi * (1.0 - ecc**2.0)

        xPos = (latus * math.cos(trueAnom)) / (1. + ecc * math.cos(trueAnom))
        yPos = (latus * math.sin(trueAnom)) / (1. + ecc * math.cos(trueAnom))
        xVel = -math.sqrt(mu/latus) * math.sin(trueAnom)
        yVel = math.sqrt(mu/latus) * (ecc + math.cos(trueAnom))


        rotMat = sl.genEulerRot(-nodeArg, -incl, -periArg,'3 1 3')

        posVec = np.dot(rotMat, np.array([[xPos],[yPos],[0.0]]))
        velVec = np.dot(rotMat, np.array([[xVel],[yVel],[0.0]]))
        posVec = np.reshape(posVec, [3,])
        velVec = np.reshape(velVec, [3,])
        return posVec, velVec
    else:
        hyperAnom = mean2hyper(meanAnom, ecc)
        trueAnom = hyper2true(hyperAnom, ecc)
        latus = semi * (1.0 - ecc ** 2.0)

        xPos = (latus * math.cos(trueAnom)) / (1. + ecc * math.cos(trueAnom))
        yPos = (latus * math.sin(trueAnom)) / (1. + ecc * math.cos(trueAnom))
        xVel = -math.sqrt(mu / latus) * math.sin(trueAnom)
        yVel = math.sqrt(mu / latus) * (ecc + math.cos(trueAnom))

        rotMat = sl.genEulerRot(-nodeArg, -incl, -periArg, '3 1 3')

        posVec = np.dot(rotMat, np.array([[xPos], [yPos], [0.0]]))
        velVec = np.dot(rotMat, np.array([[xVel], [yVel], [0.0]]))
        posVec = np.reshape(posVec, [3, ])
        velVec = np.reshape(velVec, [3, ])
        return posVec, velVec

def rv2energy(posVec, velVec, mu):
    posNorm = np.linalg.norm(posVec)
    velNorm = np.linalg.norm(velVec)

    energy = 0.5 * velNorm**2 - mu/posNorm
    return energy


def flightPathAngle(pos,vel):
    #pathAng = math.atan2(ecc*math.sin(trueAnom),1+ecc*math.cos(trueAnom))
    pathAng = math.acos(np.linalg.norm(np.cross(pos,vel)) / (np.linalg.norm(pos) * np.linalg.norm(vel)))
    return pathAng

def circleVel(planetAlt, mu):
    circVel = math.sqrt(mu/planetAlt)
    return circVel

def velCalc(currentRadius,semi,mu):
    vel = math.sqrt(mu*(2/currentRadius - 1/semi))
    return vel

def kepOrbitProp(inputType,inputArray,outputType,t,mu):
    #   Inertial pos/vel input type
    if inputType == 'rv':
        init_a, init_e, init_i, init_O, init_o, init_M = rv2orbel(inputArray[0:3],inputArray[3:],mu)
    #   Classical orbital elements input type
    elif inputType == 'classical':
        init_a = inputArray[0]
        init_e = inputArray[1]
        init_i = inputArray[2]
        init_O = inputArray[3]
        init_o = inputArray[4]
        init_M= inputArray[5]

    #Compute mean motion
    meanMotion = meanMotionComp(init_a,mu)

    #Update mean anomaly using Kepler's equation
    final_M = init_M + meanMotion * t

    if outputType == 'rv':
        #   Convert that into rv form
        final_pos, final_vel = orbel2rv(init_a, init_e, init_i, init_O, init_o, final_M, mu)

        final_pos = np.reshape(final_pos,[3,])
        final_vel = np.reshape(final_vel,[3,])
        final_rv = np.vstack([final_pos, final_vel])
        final_rv = np.reshape(final_rv, [6,])
        return final_rv

    elif outputType == 'classical':
        return init_a, init_e, init_i, init_O, init_o, final_M

def param2axis(semiparam, ecc):
    semimajor = semiparam / (1 - ecc**2)
    return semimajor

def hohmanTransfer(r_initial, r_final, a_initial, a_final, mu):
    #   Compute transfer orbit semimajor axis
    a_trans = (r_initial + r_final) / 2.0
    #print "Transit orbit SMA:", a_trans, "m"
    #   Find initial velocity using the initial position, semimajor axis velocity calculator
    v_initial = velCalc(r_initial, a_initial, mu)
    #   Ditto for the final velocity
    v_final = velCalc(r_final, a_final, mu)
    #   Ditto for the intermediate velocities, i.e. the velocities at the beginning and end of the transfer
    v_trans_a = math.sqrt((2*mu)/r_initial - mu/a_trans)
    v_trans_b = math.sqrt((2*mu)/r_final - mu/a_trans)

    #   The delta-v values are the difference between the initial circular orbit and the initial transfer ellipse,
    #   and the transfer ellipse apoapse vel and the desired final circular orbit velocity
    dv_1 = v_trans_a - v_initial
    dv_2 = v_final - v_trans_b
    dv_total = abs(dv_1) + abs(dv_2)
    #   It takes half an orbit period to make the transfer happen.
    trans_time = math.pi * math.sqrt(a_trans**3.0 / mu)
    return dv_1, dv_2, v_trans_a, v_trans_b, dv_total, trans_time

def eci2ecef(eci_pos, theta_GST):
    #   First, force the position vector into column vector form:
    np.reshape(eci_pos,[3,1])
    #   Next, multiply the position vector by the 3-rotation matrix of the GST angle:
    ecef_pos = np.dot(rot3(theta_GST), eci_pos)
    return ecef_pos

def eci2rsw(eci_pos, eci_vel):
    radialVector = eci_pos / np.linalg.norm(eci_pos)
    normalVector = np.cross(eci_pos,eci_vel,axis=0,)
    normalVector = normalVector / np.linalg.norm(normalVector)
    velVector = -np.cross(radialVector, normalVector,axis=0)
    ##print radialVector
    ##print velVector
    ##print normalVector
    rotMat = np.array([[radialVector],[velVector],[normalVector]])

    rotMat = np.reshape(rotMat,[3,3])
    ##print rotMat
    return rotMat

def eci2recthill(chief_pos, chief_vel, dep_pos, dep_vel, mu):
    chief_semi = -mu / (2.0 * (np.linalg.norm(chief_vel)**2.0/2.0 - mu/np.linalg.norm(chief_pos)))
    #print "Computed chief SMA:", chief_semi
    chief_meanmot = meanMotionComp(chief_semi, mu)
    #print "Computed chief Mean Motion:", chief_meanmot
    eci_rel_pos = -chief_pos + dep_pos
    eci_rel_vel = -chief_vel + dep_vel
    rotMat = eci2rsw(chief_pos,chief_vel)

    posNorm = np.linalg.norm(chief_pos)
    velNorm = np.linalg.norm(chief_vel)
    angVelVec = np.cross(chief_pos, chief_vel,axis=0)
    angVelNorm = np.linalg.norm(angVelVec)
    semi = (2/posNorm - (velNorm**2)/mu)**-1
    eccVec = np.cross(chief_vel, angVelVec,axis=0) / mu - chief_pos / posNorm
    chiefEcc = np.linalg.norm(eccVec)

    df_init = (chief_meanmot * chief_semi ** 2.0) / np.linalg.norm(posNorm) ** 2.0 * (math.sqrt(1.0 - chiefEcc**2.0))
    #NEED TO SUBTRACT DEPUTY RATE
    rate_vec = (np.array([0,0, -df_init])) # Written in Hill frame already

    hill_rel_pos = rotMat.dot(eci_rel_pos)
    hill_rel_vel = rotMat.dot(eci_rel_vel) + np.cross((rate_vec), hill_rel_pos)

    return hill_rel_pos, hill_rel_vel

def recthill2eci(rel_pos, rel_vel, chief_pos, chief_vel, mu):
    chief_semi = -mu / (2.0 * (np.linalg.norm(chief_vel) ** 2.0 / 2.0 - mu / np.linalg.norm(chief_pos)))
    chief_meanmot = meanMotionComp(chief_semi, mu)
    rotMat = eci2rsw(chief_pos, chief_vel)

    posNorm = np.linalg.norm(chief_pos)
    velNorm = np.linalg.norm(chief_vel)
    angVelVec = np.cross(chief_pos, chief_vel, axis=0)
    angVelNorm = np.linalg.norm(angVelVec)
    semi = (2 / posNorm - (velNorm ** 2) / mu) ** -1
    eccVec = np.cross(chief_vel, angVelVec, axis=0) / mu - chief_pos / posNorm
    chiefEcc = np.linalg.norm(eccVec)

    df_init = (chief_meanmot * chief_semi ** 2.0) / np.linalg.norm(posNorm) ** 2.0 * (math.sqrt(1.0 - chiefEcc ** 2.0))

    n2hill = eci2rsw(chief_pos, chief_vel)
    hill2n = np.transpose(n2hill)

    rate_vec = hill2n.dot(np.array([0, 0, df_init]))

    eci_rel_pos = hill2n.dot(rel_pos)
    eci_rel_vel = hill2n.dot(rel_vel) + np.cross(rate_vec, eci_rel_pos)

    dep_pos = chief_pos + eci_rel_pos
    dep_vel = chief_vel + eci_rel_vel

    return dep_pos, dep_vel

class relMotOptions():
    def __init__(self, mu):
        self.mu = mu

def relMotPropFcn(t,state,relMotOptions):

    mu = relMotOptions.mu

    x = state[0]
    y = state[1]
    z = state[2]
    dx = state[3]
    dy = state[4]
    dz = state[5]
    f = state[6]
    df = state[7]
    rc = state[8]
    drc = state[9]

    rd = math.sqrt((rc+x)**2.0 + y**2.0 + z**2.0)

    ddx = 2.* df * (dy - y*(drc/rc)) + x * df**2.0 + mu/(rc**2.0) - mu/(rd**3.0) *(rc+x)
    ddy = -2.*df * (dx - x*(drc/rc)) + y*df**2.0 - mu/rd**3.0 * y
    ddz = -mu/(rd**3.0) * z
    ddf = -2. * drc / rc * df
    ddrc = rc * df**2.0 -mu/(rc**2.0)


    deltaState = np.array([dx, dy, dz, ddx, ddy, ddz, df, ddf, drc, ddrc])
    return deltaState

class curvMotOptions():
    def __init__(self, rc, n):
        self.rc = rc
        self.n = n

def curvRelMotPropFcn(t,state,relMotOptions):

    rc = relMotOptions.rc
    n = relMotOptions.n

    dr = state[0]
    dtheta = state[1]
    z = state[2]
    ddr = state[3]
    ddtheta = state[4]
    dz = state[5]

    dddr = 2.*n*rc*ddtheta + 3.*n**2.0 * dr
    dddtheta = -2.*n*ddr/rc
    ddz = -n**2.0 * z

    deltaState = np.array([ddr, ddtheta, dz, dddr, dddtheta, ddz])
    return deltaState

def hcwPropFcn(t,state,relMotOptions):
    rc = relMotOptions.rc
    n = relMotOptions.n

    x = state[0]
    y = state[1]
    z = state[2]
    dx = state[3]
    dy = state[4]
    dz = state[5]

    ddx = (2. * n * dy) + (3. * n ** 2.0 * x) #- (0.5 * Bd * Pd * n * rc) * dx
    ddy = -2. * n * dx  #+ (n**2.0 * rc**2.0) * 0.5 * (Bc * Pc - Bd * Pd) - Bd*Pd*n*rc*dy
    ddz = -n ** 2.0 * z #- 0.5 * Bd * Pd * rc * n * dz

    deltaState = np.array([dx,dy,dz,ddx,ddy,ddz])
    return deltaState

class dragMotOptions():
    def __init__(self, rc, n, Bc, Bd, Pc, Pd):
        self.rc = rc
        self.n = n
        self.Bc = Bc
        self.Bd = Bd
        self.Pc = Pc
        self.Pd = Pd

def hcwDragPropFcn(t,state,relMotOptions):
    rc = relMotOptions.rc
    n = relMotOptions.n
    Bc = relMotOptions.Bc
    Bd = relMotOptions.Bd
    Pc = relMotOptions.Pc
    Pd = relMotOptions.Pd


    x = state[0]
    y = state[1]
    z = state[2]
    dx = state[3]
    dy = state[4]
    dz = state[5]

    ddx = (2. * n * dy) + (3. * n ** 2.0 * x) - (0.5 * Bd * Pd * n * rc) * dx
    ddy = -2. * n * dx + (n**2.0 * rc**2.0) * 0.5 * (Bc * Pc - Bd * Pd) - Bd*Pd*n*rc*dy
    ddz = -n ** 2.0 * z - 0.5 * Bd * Pd * rc * n * dz

    deltaState = np.array([dx, dy, dz, ddx, ddy, ddz])
    return deltaState


def curvMotionProp(relState, mu, time):
    init_vec = relState
    init_vec = np.reshape(init_vec, [6,])

    intOptions = relMotOptions(mu)

    final_vec = sl.rk4(curvRelMotPropFcn,time[0],time[1]-time[0],init_vec, intOptions)
    return final_vec

def relMotionProp(relState, mu, time):
    init_vec = relState
    init_vec = np.reshape(init_vec, [10,])

    intOptions = relMotOptions(mu)

    final_vec = sl.rk4(relMotPropFcn,time[0],time[1]-time[0],init_vec, intOptions)
    return final_vec

def hcwProp(relState, n, rc, time):
    init_vec = relState
    init_vec = np.reshape(init_vec, [6, ])

    intOptions = curvMotOptions(rc, n)

    final_vec = sl.rk4(hcwPropFcn, time[0], time[1] - time[0], init_vec, intOptions)
    return final_vec

def dragHcwProp(relState, n, rc, Bc, Bd, Pc, Pd, time):
    init_vec = relState
    init_vec = np.reshape(init_vec, [6, ])
    intOptions = dragMotOptions(rc, n, Bc, Bd, Pc, Pd)

    final_vec = sl.rk4(hcwDragPropFcn, time[0], time[1] - time[0], init_vec, intOptions)
    return final_vec

def recthill2relOE(chief_pos, chief_vel, rel_vec, mu):
    semi, ecc, incl, periArg, nodeArg, meanAnom = rv2orbel(chief_pos, chief_vel, mu)

    q1 = ecc * math.sin(periArg)
    q2 = ecc * math.cos(periArg)

    eccAnom = mean2ecc(meanAnom,ecc)
    trueAnom = ecc2true(eccAnom,ecc)
    theta = trueAnom + periArg

    chief_radius = np.linalg.norm(chief_pos)


    ang_mom = np.linalg.norm(np.cross(chief_pos, chief_vel))
    latus = semi * (1.0 - q1**2.0 -q2**2.0)
    Vr = ang_mom/latus * (q1*math.sin(theta) - q2 * math.cos(theta))
    Vt = ang_mom/latus * (1 + q1*math.cos(theta) + q2 * math.sin(theta))

    alpha = semi/chief_radius
    rho = chief_radius/latus
    nu = Vr/Vt
    k1 = alpha * (1./rho - 1.)
    k2 = alpha * nu**2.0 * 1./rho

    cottheta = 1./math.tan(theta)

    a11 = 2.*alpha*(2. + 3.*k1 * 2.*k2)
    a12 = -2. * alpha * nu * (1. + 2.*k1 + k2)
    a13 = 0.0
    a14 = (1./Vt) * (2.* alpha**2.0 * nu * latus)
    a15 = (2.*semi/Vt) * (1. + 2. * k1 + k2)
    a16 = 0.0
    a21 = 0.0
    a22 = 1./chief_radius
    a23 = cot(incl)/chief_radius * ( math.cos(theta) + nu * math.sin(theta))
    a24 = 0.0
    a25 = 0.0
    a26 = -math.sin(theta) * cot(incl) / Vt
    a31 = 0.0
    a32 = 0.0
    a33 = (math.sin(theta) - nu*math.cos(theta))/chief_radius
    a34 = 0.0
    a35 = 0.0
    a36 = math.cos(theta)/Vt
    a41 = 1.0/(rho*chief_radius) * (3. * math.cos(theta) +2.*nu * math.sin(theta))
    a42 = -1/chief_radius * ((nu**2.0/rho)*math.cos(theta) + q1*math.sin(2.*theta) - q2 * math.cos(2.*theta))
    a43 = -(q2*cot(incl))/chief_radius * (math.cos(theta) + nu*math.sin(theta))
    a44 = math.sin(theta)/(rho*Vt)
    a45 = (1./(rho*Vt))*(2.*math.cos(theta) + nu*math.sin(theta))
    a46 = (q2*cot(incl)*math.sin(theta))/Vt
    a51 = (1./(rho*chief_radius)) * (3. * math.sin(theta) - 2. *nu*math.cos(theta))
    a52 = (1./chief_radius) * (nu**2.0/rho *math.sin(theta) + q2*math.sin(2.*theta) + q1*math.cos(2.*theta))
    a53 = (q1*cot(incl))/chief_radius * (math.cos(theta) + nu*math.sin(theta))
    a54 = -math.cos(theta)/(rho*Vt)
    a55 = 1.0/(rho*Vt) * ( 2. * math.sin(theta) - nu*math.cos(theta))
    a56 = -(q1*cot(incl) * math.sin(theta))/Vt
    a61 = 0.0
    a62 = 0.0
    a63 = -(math.cos(theta)+nu*math.sin(theta))/(chief_radius*math.sin(incl))
    a64 = 0.0
    a65 = 0.0
    a66 = math.sin(theta)/(Vt * math.sin(incl))

    A_inv = np.array([[a11, a12, a13, a14, a15, a16],
                      [a21, a22, a23, a24, a25, a26],
                      [a31, a32, a33, a34, a35, a36],
                      [a41, a42, a43, a44, a45, a46],
                      [a51, a52, a53, a54, a55, a56],
                      [a61, a62, a63, a64, a65, a66]])

    return A_inv, A_inv.dot(rel_vec)

def relOE2recthill(chief_pos, chief_vel, rel_oes, mu):
    semi, ecc, incl, periArg, nodeArg, meanAnom = rv2orbel(chief_pos, chief_vel, mu)

    da = rel_oes[0]
    dtheta = rel_oes[1]
    dincl = rel_oes[2]
    dq1 = rel_oes[3]
    dq2 = rel_oes[4]
    draan = rel_oes[5]

    q1 = ecc * math.sin(periArg)
    q2 = ecc * math.cos(periArg)

    eccAnom = mean2ecc(meanAnom,ecc)
    trueAnom = ecc2true(eccAnom,ecc)
    theta = trueAnom + periArg

    chief_radius = np.linalg.norm(chief_pos)


    ang_mom = np.linalg.norm(np.cross(chief_pos, chief_vel))
    latus = semi * (1.0 -q1**2.0 - q2**2.0)
    Vr = ang_mom/latus * (q1*math.sin(theta) - q2 * math.cos(theta))
    Vt = ang_mom/latus * (1 + q1*math.cos(theta) + q2 * math.sin(theta))
    alpha = semi/chief_radius
    rho = chief_radius/latus
    nu = Vr/Vt
    k1 = alpha * (1./rho - 1.)
    k2 = alpha * nu**2.0 * 1./rho

    a11 = chief_radius/semi
    a12 = Vr/Vt * chief_radius
    a13 = 0.0
    a14 = -(chief_radius)/latus * (2. * semi * q1 + chief_radius * math.cos(theta))
    a15 = -(chief_radius)/latus * (2. * semi * q2 + chief_radius * math.sin(theta))
    a16 = 0.0
    a21 = 0.0
    a22 = chief_radius
    a23 = 0.0
    a24 = 0.0
    a25 = 0.0
    a26 = math.cos(incl)
    a31 = 0.0
    a32 = 0.0
    a33 = chief_radius * math.sin(theta)
    a34 = 0.0
    a35 = 0.0
    a36 = -math.cos(theta) * math.sin(incl) * chief_radius
    a41 = -Vr / (2*semi)
    a42 = (1/chief_radius - 1/latus) * ang_mom
    a43 = 0.0
    a44 = 1/latus * (Vr * semi * q1 + ang_mom * math.sin(theta))
    a45 = 1/latus * (Vr * semi * q2 - ang_mom * math.cos(theta))
    a46 = 0.0
    a51 = (-3. * Vt)/(2. * semi)
    a52 = -Vr
    a53 = 0.0
    a54 = (3. * Vt * semi * q1 + 2. * ang_mom * math.cos(theta)) * 1./latus
    a55 = (2. * Vt * semi * q2 + 2. * ang_mom * math.sin(theta)) * 1./latus
    a56 = Vr * math.cos(incl)
    a61 = 0.0
    a62 = 0.0
    a63 = (Vt * math.cos(theta) + Vr * math.sin(theta))
    a64 = 0.0
    a65 = 0.0
    a66 = (Vt * math.sin(theta) - Vr * math.cos(theta)) * math.sin(incl)

    A = np.array([[a11, a12, a13, a14, a15, a16],
                  [a21, a22, a23, a24, a25, a26],
                  [a31, a32, a33, a34, a35, a36],
                  [a41, a42, a43, a44, a45, a46],
                  [a51, a52, a53, a54, a55, a56],
                  [a61, a62, a63, a64, a65, a66]])

    return A, A.dot(rel_oes)

def ecef2lla(ecef_pos):
    r = np.linalg.norm(ecef_pos)
    phi = math.asin(ecef_pos[2]/r)
    #lam = math.acos(ecef_pos[0] / (r*math.cos(phi)))
    lam = math.atan2(ecef_pos[1],ecef_pos[0])
    return phi, lam, r


def ecef2eci(ecef_pos, theta_GST):
    np.reshape(ecef_pos,[3,1])
    eci_pos = np.dot(np.linalg.inv(rot3(theta_GST)), ecef_pos)
    return eci_pos


def lla2ecef(phi, lam, r):
    ecef_pos = np.zeros([3,1])
    ecef_pos[0] = r*math.cos(phi) * math.cos(lam)
    ecef_pos[1] = r*math.cos(phi) * math.sin(lam)
    ecef_pos[2] = r*math.sin(phi)
    return ecef_pos


def ecef2topo(sat_ecef, station_lat, station_long, station_alt):
    lat_rot = rot2((math.pi/2) - station_lat)
    lon_rot = rot3(station_long)
    station_ecef = lla2ecef(station_lat, station_long, station_alt)
    rel_ecef = sat_ecef - station_ecef
    rel_sez = np.dot(np.dot(lat_rot, lon_rot), rel_ecef)
    rel_range = np.linalg.norm(rel_ecef)
    rel_el = math.asin(rel_sez[2] / rel_range)
    rel_az = math.atan2(rel_sez[1], -rel_sez[0])


    return rel_range, rel_az, rel_el, rel_sez

def tleDecoder(line_1, line_2):
    line_1_decoded = line_1.split(" ")
    line_2_decoded = line_2.split(" ")

    sat_number = int(line_1[2:7])
    epoch_year = float(line_1[18:20])
    epoch_day = float(line_1[20:32])
    incl = float(line_2[8:16]) * math.pi/180.0
    raan = float(line_2[17:25])* math.pi/180.0
    ecc = float("0."+line_2[26:33])
    peri = float(line_2[34:42])* math.pi/180.0
    mean_anom = float(line_2[43:51])* math.pi/180.0
    mean_motion = float(line_2[52:63]) * 7.27221e-5

    return sat_number, epoch_year, epoch_day, mean_motion, ecc, incl, raan, peri, mean_anom

def doy2date(year, doy):
    #   This code steals lots of math from JD2Date, which Vallato wrote.
    #   First, find the number of decimal days and convert it to hours.
    hours_raw = (doy % 1) * 24.0
    #   The hour is the integer part of hours_raw.
    hours = math.trunc(hours_raw)
    #   Next, take the decimal part of hours_raw and convert it to minutes.
    minutes_raw = (hours_raw % 1) * 60.0
    #   Rinse and repeat for minutes/seconds.
    minutes = math.trunc(minutes_raw)
    seconds = (minutes_raw % 1) * 60.0
    #   Set up a vector of month lengths and month names
    month_length = [31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0]
    month_vec = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    #   Check for leap years and adjust february accordingly
    if year % 4 == 0:
        #print "it's a leap year!"
        month_length[1] = 29
    day_of_year = math.trunc(doy)
    #print day_of_year
    day_sum = 0
    ind = 0
    #   Loop over the month lengths until the sum is greater than the current day; then, break and subtract off the last month.
    while day_sum < day_of_year:
        day_sum = day_sum + month_length[ind]
        if day_sum > day_of_year:
            day_sum = day_sum - month_length[ind]
            break

        ind = ind+1
    #   Congrats, you found the month.
    month = month_vec[ind]
    #print month
    date = math.trunc(doy - day_sum)
    #   Give the user back their variables.
    return year, month, date, hours, minutes, seconds

def jd2date(JD):
    #   This entire algorithm is a direct implementation of Vallado.
    #   Calculate number of Julian Years since 1900
    T_1900 = (JD-2415019.5)/365.25
    year = 1900.0 + math.trunc(T_1900)
    #print year
    leap_years = math.trunc((year-1900.0-1.90)*0.25)
    #print "no leap years",leap_years
    leap_years = 29
    days = (JD-2415019.5)-((year-1900.0)*(365.0)+leap_years)
    #print days
    if days < 1.0: #    If it hasn't even been a day, back the year off 1 and make sure we re-account for days/leap years.
        year = year-1
        leap_years= math.trunc((year-1900.0-1.90)*0.25)
        days = (JD-2415019.5)-((year-1900)*(365.0)+leap_years)

    #   Define the months and their lengths. This took ages.
    month_length = [31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0 , 30.0, 31.0, 30.0, 31.0]
    month_vec = ["January","February","March","April","May","June","July","August","September","October","November","December"]

    #   If the year is divisible by four...
    if year % 4 == 0:
        #print "it's a leap year!"
        month_length[1] = 29    #change the length of february
    #   The day of the year is the integer day of the year.
    day_of_year = math.trunc(days)
    #print day_of_year
    day_sum = 0
    ind = 0
    #   Add up the lengths of months until we get to the current day of year. If we exceed it, we've gone too far, so back it off one and that's our month.n e
    while day_sum < day_of_year:
        day_sum = day_sum + month_length[ind]
        if day_sum > day_of_year:
            day_sum = day_sum - month_length[ind]
            break

        ind = ind+1
    #   Figure out the name of the month. No, computer, the index won't suffice.
    month = month_vec[ind]
    #print month
    #   The day of the month is basically the day of the year minus the number of days leading up to the current month
    day = day_of_year - day_sum
    #print day_of_year
    #print day
    #   Look at the remaining decimal days and convert downward to hours, minutes and seconds.
    tau = (days - day_of_year) * 24.0
    hour = math.trunc(tau)
    minute = math.trunc((tau - hour) * 60.0)
    second = (tau - hour - minute/60.0)*60.0

    return year, month, day, hour, minute, second

def date2jd(year , month , day , hour , minute , sec ):
    #   This is a direct implementation of a formula from Vallado. It effectively successively converts smaller and smaller units of time into days and adds it to the counter.
    JD = 367.0*year - int((7.0 * (year+int((month + 9.0)/12.0))) / 4.0) + int(275.0 * month / 9.0) + day + 1721013.5 + ((sec / 60.0 + minute)/60.0 + hour)/24.0
    return JD

def ut2theta(t_ut):
    #   Implement the equation from Vallado. Note that t_ut is the number of Julian Centuries since J2000. (Why, Vallado, why!?)
    theta_gmst_0 = 100.4606184 + 36000.77005361 * t_ut + 0.00038793*t_ut**2.0 - 2.6e-8 * t_ut**3.0
    theta_gmst_0 = theta_gmst_0 * math.pi/180.0
    theta_gmst_0 = theta_gmst_0
    return theta_gmst_0

def theta02theta(theta0, UT):
    omega_c = 7.2921158553e-5 # Value in rad/s from back of vallado
    theta_gmst = theta0 + omega_c * UT #    Equation given in Vallado for rotation of earth at any UTC
    return theta_gmst

def boundedRelOrbitCalc(chief_pos, chief_vel, offset, flagType, mu):
    semi, ecc, incl, periArg, nodeArg, meanAnom = rv2orbel(chief_pos, chief_vel, mu)

    meanMot = meanMotionComp(semi,mu)
    bounding_val = (-meanMot*(2.+ecc))/(math.sqrt((1.+ecc)*(1.-ecc)**3.0))

    if flagType == "x0":
        output = offset * bounding_val
    elif flagType == "dy0":
        output = offset / bounding_val
    else:
        print "Warning, bad flag set."
        output = 9999
    return output

def controlSensitivityOe(oeSet, mu, req):
    req = req/1000.
    mu = mu / 1e9
    a = oeSet[0]*req
    e = oeSet[1]
    i = oeSet[2]
    raan = oeSet[3]
    peri = oeSet[4]
    meanAnom = oeSet[5]
    n = meanMotionComp(a, mu)

    stateVec = orbelConverter('classical','rv', np.array([a,e,i,raan,peri,meanAnom]), mu)
    rVec = stateVec[0:3]

    #   Calculation of other necessary variables
    p = a * (1-e**2.0)
    h = math.sqrt(mu*p)
    b = a * math.sqrt(1.0 - e**2.0)

    eccanom = mean2ecc(meanAnom,e)
    f = ecc2true(eccanom, e)

    r = p / (1 + e * math.cos(f))
    theta = peri+f
    eta = b/a

    B11 = (2.* (a**2.0) * e * math.sin(f))/(h*req)
    B12 = (2.* (a**2.0) * p)/(h*r*req)
    B21 = (p*math.sin(f))/h
    B22 = ((p+r)*math.cos(f)+r*e)/h
    B33 = (r*math.cos(theta))/h
    B43 = (r*math.sin(theta))/(h*math.sin(i))
    B51 = -(p*math.cos(f))/(h*e)
    B52 = ((p+r) * math.sin(f))/(h*e)
    B53 = -(r*math.sin(theta)*math.cos(i) )/(h*math.sin(i))
    B61 = (eta*(p*math.cos(f) - 2.*r*e))/(h*e)
    B62 = - (eta*(p+r)*math.sin(f))/(h*e)

    B = np.array([[B11, B12, 0],
                  [B21, B22, 0],
                  [0,   0,  B33],
                  [0,   0,  B43],
                  [B51, B52, B53],
                  [B61, B62, 0]])

    return B

class oeOptions():
    def __init__(self, mu, req, j2):
        self.mu = mu
        self.req = req
        self.j2 = j2
        self.u = np.zeros([3,])
        self.P = np.zeros([3,])
        self.targOeSet = np.zeros([6,])
        return

def oePropFcn(t, y, oeOptions):
    n = meanMotionComp(y[0]*oeOptions.req, oeOptions.mu)

    A = np.zeros([6,])
    A[5] = n

    y_dot = A
    return y_dot

def oePropFcn2(t, y, oeOptions):
    n = meanMotionComp(y[0], oeOptions.mu)

    A = np.zeros([6,])
    A[5] = n

    y_dot = A
    return y_dot

def clOePropFcn(t, y, oeOptions):
    n = meanMotionComp(y[0]*oeOptions.req, oeOptions.mu)

    A = np.zeros([6,])
    A[5] = n

    deltaOE = y - oeOptions.targOeSet

    B = controlSensitivityOe(y, oeOptions.mu, oeOptions.req)
    u = -np.diag(oeOptions.P).dot(np.transpose(B).dot(deltaOE))

    y_dot = A + B.dot(u)

    return y_dot

def eulerInt(fun, t0, dt, y0, opts):
    #sl.rk4(kl.clOePropFcn, tvec[ind-1], (tvec[ind]-tvec[ind-1]), depOrbel[:,ind-1], depOptions )
    y1 = y0 + (fun(t0,y0,opts)*dt)
    return y1






