import math
import numpy as np
import scipy as sci
import keplerLib as kl
import perturbationLib as pl

def targetRateCalc(target_semi, mu):
    target_ang_rate = math.sqrt(mu/target_semi**3)
    return target_ang_rate

def phaseTime(orbit_count, target_rate, theta):
    phase_time = (orbit_count * 2 * math.pi + theta)/target_rate
    return phase_time

def orbWait(phase_time, target_rate, theta):
    orbit_count = (phase_time * target_rate - theta)/(2.0 * math.pi)

def phaseSemi(orbit_count, mu, phase_time):
    phase_semi = (mu*(phase_time / (orbit_count*2.0*math.pi))**2.0)**(1.0/3.0)
    return phase_semi

def cwTransfer(initialState,targNu, time):
    x0 = initialState[0]
    y0 = initialState[1]
    z0 = initialState[2]
    dx0 = initialState[3]
    dy0 = initialState[4]
    dz0 = initialState[5]

    y_dot_num = (6.0*x0*(targNu*time-math.sin(targNu*time))-y0)*targNu*math.sin(targNu*time) - 2.0*targNu*x0*(4.0-3.0*math.cos(targNu*time))*(1-math.cos(targNu*time))
    y_dot_den = (4.0*math.sin(targNu*time) - 3.0*targNu*time)*math.sin(targNu*time) + 4.0*(1.0 - math.cos(targNu*time))**2.0
    y_dot = y_dot_num / y_dot_den

    x_dot_num = (targNu*x0*(4.0-3.0*math.cos(targNu*time)) + 2.0*(1.0 - math.cos(targNu*time))*y_dot)
    x_dot_den = math.sin(targNu*time)
    x_dot = -x_dot_num / x_dot_den

    z_dot = -z0*targNu*(1/math.tan(targNu*time))

    vel_vec = np.array([[x_dot],[y_dot],[z_dot]])
    return vel_vec

def cwPropagate(initialState, targNu,time):
    x0 = initialState[0]
    y0 = initialState[1]
    z0 = initialState[2]
    dx0 = initialState[3]
    dy0 = initialState[4]
    dz0 = initialState[5]

    x = dx0/targNu * math.sin(targNu * time) - (3.* x0 + 2.*dy0/targNu) * math.cos(targNu*time) + (4.*x0 + 2.*dy0/targNu)
    y =  2.* (3.* x0 + 2.*dy0/targNu) * math.sin(targNu*time)+ 2.*dx0/targNu * math.cos(targNu*time) - (6.*targNu*x0 + 3.*dy0) *time + (y0 - 2.*dx0/targNu)
    z = z0 * math.cos(targNu*time)+dz0/targNu * math.sin(targNu*time)

    dx = dx0*math.cos(targNu*time) + (3.0*targNu*x0+(2.0*dy0))*math.sin(targNu*time)
    dy = (6.0*targNu*x0+(4.0*dy0))*math.cos(targNu*time) - 2.*dx0*math.sin(targNu*time)-(6.*targNu*x0+3.*dy0)
    dz = -z0*targNu*math.sin(targNu*time) + dz0*math.cos(targNu*time)

    finalPos = np.array([x,y,z])
    finalVel = np.array([dx,dy,dz])

    return finalPos, finalVel

def inclOnlyTransfer(v_init, flight_path_ang, incl_change):
    dv = 2.0*v_init*math.cos(flight_path_ang) * math.sin(incl_change / 2.0)
    return dv

def soiCalc(planetName):

    sunMass = 1.9891e30
    if planetName == "Earth":
        orbitRadius = 149598023.0e3
        planetMass = 5.9742e24
    elif planetName == "EarthMoon":
        orbitRadius = 149598023.0e3
        planetMass = 5.9742e24 + 7.3483e22
    elif planetName == "Mars":
        orbitRadius = 227939186.0e3
        planetMass = 6.4191e23
    elif planetName=="Jupiter":
        orbitRadius =778298361.0e3
        planetMass = 1.8988e27
    elif planetName == "Saturn":
        orbitRadius = 1429394133e3
        planetMass = 5.685e26
    elif planetName == "Asteroid":
        orbitRadius = 3 * 149598023.0e3
        planetMass = 1.0e15
    soiRadius = (planetMass / sunMass)**(2.0/5.0) * orbitRadius
    return soiRadius

def gravAssistAngle(mu,rp,excessVel,initAngle, planetHelioVel, satPos):
    excessVel = abs(excessVel)
    invEcc = (1.0 / ( 1.0 + (excessVel**2.0 * rp)/(mu)))
    #print invEcc
    ecc = 1.0/invEcc
    turnAngle = math.asin(invEcc) * 2



    initVel = np.array([[excessVel],[0.0],[0.0]])
    initVel = np.dot(kl.rot3(initAngle),initVel)
    if satPos == "Behind":
        turnedVel = np.dot(kl.rot3(turnAngle),initVel)
    elif satPos == "Front":
        turnedVel = np.dot(kl.rot3(turnAngle),-initVel)

    planetVel = np.array([[planetHelioVel],[0.0],[0.0]])
    finalVel = turnedVel + planetVel
    finalVelMag = np.linalg.norm(finalVel)
    turnedVel = np.reshape(turnedVel, [3,])

    return turnAngle, turnedVel, finalVelMag

def raanOnlyTransfer(deltaRAAN, initIncl, initVel):

    theta = math.acos( math.cos(initIncl)**2.0 + math.sin(initIncl)**2.0 * math.cos(deltaRAAN))
    deltaV = 2.0 * initVel * math.sin(theta/2)
    return deltaV

def nonlinCartesianJ2Law(chiefPos, chiefVel, depPos, depVel, K1, K2, mu):

    options = pl.odeOptions(mu, np.zeros([3,]),0.0)
    initChiefState = np.zeros([6,])
    initChiefState[0:3] = chiefPos
    initChiefState[3:] = chiefVel
    initDepState = np.zeros([6,])
    initDepState[0:3] = depPos
    initDepState[3:] = depVel

    chiefAcc = pl.j2PropModel(0, initChiefState, options)
    depAcc = pl.j2PropModel(0, initDepState, options)

    posErr = depPos - chiefPos
    velErr = depVel - chiefVel

    dynDiff = depAcc[3:] - chiefAcc[3:]

    controlOut = -dynDiff - K1.dot(posErr) - K2.dot(velErr)

    return controlOut

def clOeControl(B, P, req, desOE, currOE):

    desOE[0] = desOE[0]
    currOE[0] = currOE[0]
    deltaOE =  currOE - desOE

    #u = -P * pseudoInv.dot(np.transpose(B)).dot(np.array(deltaOE))
    u = -np.diag(P).dot(np.transpose(B).dot(deltaOE))
    #u = np.zeros([3,])
    #print "Control effort:"
    #print u
    desOE[0] = desOE[0]
    currOE[0] = currOE[0]
    return u

def impulsiveControl(oeSet, deltaOE, req, mu, burnIndex, nextBurnDv, dt):
    req = req
    mu = mu
    a = oeSet[0]
    e = oeSet[1]
    i = oeSet[2]
    raan = oeSet[3]
    peri = oeSet[4]
    meanAnom = oeSet[5] % (2. * math.pi)
    n = kl.meanMotionComp(a, mu)
    tol = 2*n/dt

    stateVec = kl.orbelConverter('classical', 'rv', np.array(oeSet), mu)
    rVec = stateVec[0:3]
    vVec = stateVec[3:]

    #   Calculation of other necessary variables
    p = a * (1 - e ** 2.0)
    h = math.sqrt(mu * p)
    b = a * math.sqrt(1.0 - e ** 2.0)

    eccanom = kl.mean2ecc(meanAnom, e)
    f = kl.ecc2true(eccanom, e)

    r = p / (1 + e * math.cos(f))
    theta = (peri + f)%2.*math.pi
    eta = b / a
    atApo = 0
    atPeri = 0
    #print "meanAnom:", meanAnom
    if meanAnom < tol and meanAnom > -tol:
        #print "Peri Burn:"
        atPeri = 1
    if meanAnom < math.pi + tol and meanAnom > math.pi - tol:
        #   youre at apoapsis. do those burns
        #print "Apo burn:"
        atApo = 1

    ##  Compute delta OEs for correction locations
    #   Compute new deltaOEs
    deltaSMA = deltaOE[0]
    deltaEcc = deltaOE[1]
    deltaIncl = deltaOE[2]
    deltaOmega = deltaOE[3]
    deltaPeri = deltaOE[4]
    deltaM = deltaOE[5]

    thetaC = math.atan2(deltaOmega * math.sin(i), deltaIncl)%(2.*math.pi)
    dv = np.zeros([3, ])
    maneuver = 0
    if theta < thetaC + tol and theta > thetaC - tol and burnIndex == 0:
        #   Do RAAN, incl correction
        print "RAAN/INCL burn:"
        dv[2] = h / r * math.sqrt(deltaIncl ** 2.0 + (deltaOmega * math.sin(i)) ** 2.0)
        maneuver = 1

    if atPeri and burnIndex == 1:
        #   you're at periapsis. Correct periapsis and M
        dv[0] = -n * a / 4 * (((1. + e)** 2.0) / eta * (deltaPeri + deltaOmega * math.cos(i)) + deltaM)
        nextBurnDv[0] = -n*a/4. * (((1. - e) ** 2.0) / eta * (deltaPeri + deltaOmega * math.cos(i)) + deltaM)

        maneuver = 1

    if atApo and burnIndex == 2:
        #   youre at apoapsis. finish the periapsis and M burns
        dv[0] = nextBurnDv[0]
        nextBurnDv = np.zeros([3,])
        maneuver = 1

    if atPeri and burnIndex == 3:
        #   you're at peri. Start the a and e corrections
        dv[1] = n*a*eta / 4. * (deltaSMA/a + deltaEcc / (1 + e))
        nextBurnDv[1] = n*a*eta / 4. * (deltaSMA / a - deltaEcc / (1 - e))
        maneuver=1

    if atApo and burnIndex == 4:
        #   you're at apo. finish the a and e corrections
        dv[1] = nextBurnDv[1]
        nextBurnDv = np.zeros([3,])
        maneuver=1

    if maneuver == 1:
        print dv
        newState = np.zeros([6, ])
        rotDv = np.transpose(kl.eci2rsw(rVec, vVec)).dot(dv)
        newState[0:3] = rVec
        newState[3:] = vVec + rotDv
        newOrbel = kl.orbelConverter('rv', 'classical', newState, mu)
        burnIndex = burnIndex + 1
        print "BURN INDEX IS NOW: ", burnIndex
        return newOrbel, dv, burnIndex, nextBurnDv
    else:
        return oeSet, dv, burnIndex, nextBurnDv

def nonlinDragLaw(chiefPos, chiefVel, depPos, depVel, K1, K2, mu, beta_c, maxBeta, minBeta):

    P_c = pl.earthExpAtmo(chiefPos)
    P_d = pl.earthExpAtmo(depPos)

    chiefDragAcc = pl.simpleDrag(beta_c, P_c, chiefVel)
    options = pl.odeOptions(mu, np.zeros([3,]), 0.0)
    initChiefState = np.zeros([6,])
    initChiefState[0:3] = chiefPos
    initChiefState[3:] = chiefVel
    initDepState = np.zeros([6,])
    initDepState[0:3] = depPos
    initDepState[3:] = depVel

    chiefAcc = pl.propModel(0, initChiefState, options)
    depAcc = pl.propModel(0, initDepState, options)

    posErr = depPos - chiefPos
    velErr = depVel - chiefVel

    dynDiff = depAcc[3:] - chiefAcc[3:] + chiefDragAcc

    depVelDir = depVel/np.linalg.norm(depVel)

    controlOut = np.diag(depVelDir).dot(-dynDiff - K1.dot(posErr) - K2.dot(velErr))
    outMag = np.linalg.norm(controlOut)
    if outMag > maxBeta:
        controlOut = controlOut * (maxBeta / outMag)
    elif outMag < maxBeta:
        controlOut = controlOut * (minBeta/outMag)

    return controlOut