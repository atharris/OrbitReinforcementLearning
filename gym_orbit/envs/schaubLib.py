from math import *
import numpy as np
import scipy as sci
import sys
import keplerLib as kl
import matplotlib.animation as animation


def backwardDiff(previous, current, timeStep):
    delta = (current - previous)/timeStep
    return delta

def rk4(fun,t0,dt,y0, funOptions):
    #INPUTS:
    # fun: function to be integrated, defined as ynew = fun(t,y)
    # t0: time at y0
    # dt: designated step size (also ref'd as 'h')
    # y0: initial conditions
    k1 = fun(t0,y0,funOptions)
    k2 = fun(  t0+dt/2.0,  y0 + dt/2.0 * k1,funOptions)
    k3 = fun(t0+dt/2.0, y0 + dt/2.0 * k2,funOptions)
    k4 = fun(t0 + dt, y0 + dt*k3,funOptions)
    y1 = y0 + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

    return y1

class rk4Options:
    def __init__(self, rateVec):
        self.rateVec = rateVec

def skew(vec):
    skewMat = np.array([[ 0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
    return skewMat

def genEulerRot(ang1, ang2, ang3, rotOrder):
    #Generates the rotation matrix BN for an euler rotation defined by rotOrder
    #INPUTS:
    #   ang1, ang2, ang3: sucessive rotations in radians
    #   rotOrder: type of Euler rotation (ex: "3 1 3")

    angleList = rotOrder.split()
    evalString = "np.dot(kl.rot" + angleList[2]+"(ang3), np.dot(kl.rot" +angleList[1]+"(ang2),kl.rot" + angleList[0] + "(ang1)))"
    rotMat = eval(evalString)
    return rotMat

def dcm2prv(dcm):
    try:
        rotAngle = acos( 0.5 * (dcm[0,0] + dcm[1,1] + dcm[2,2] - 1))
    except ValueError:
        #print "Warning: Angle is near-zero."
        rotAngle = 0.0
    try:
        rotVector = 1 / (2.0*sin(rotAngle)) * np.array([dcm[1,2] - dcm[2,1], dcm[2,0] - dcm[0,2], dcm[0,1] - dcm[1,0]])
    except ZeroDivisionError:
        #print "Warning: PRV Angle zero."
        rotVector = np.array([1,0,0])
    return rotAngle, rotVector

def prv2dcm(prvAngle, prvVector):
    cphi = cos(prvAngle)
    sphi = sin(prvAngle)
    sigma = 1.0 - cphi

    e1 = prvVector[0]
    e2 = prvVector[1]
    e3 = prvVector[2]
    print np.linalg.norm(prvVector)
    dcm = np.array([ [(e1**2.0) * sigma + cphi, e1*e2*sigma + e3*sphi, e1*e3*sigma - e2*sphi],
                     [e2*e1*sigma - e3*sphi, (e2**2.0) * sigma + cphi, e2*e3*sigma + e1*sphi],
                     [e3*e1 * sigma + e2*sphi, e3*e2*sigma - e1*sphi, (e3**2.0 )* sigma + cphi]])
    return dcm

def prv2quat(prvAngle, prvVector):
    b0 = cos(prvAngle/2.0)
    sinAngle = sin(prvAngle/2.0)
    b1 = prvVector[0] * sinAngle
    b2 = prvVector[1] * sinAngle
    b3 = prvVector[2] * sinAngle

    quat = np.array([b0, b1, b2, b3])
    return quat

def dcm2quat(dcm):
    dcmTrace = np.trace(dcm)
    b_test = np.zeros([4])
    quat = np.zeros([4])
    b_test[0] = 0.25 * (1.0 + dcmTrace)
    b_test[1] = 0.25 * (1.0 + 2*dcm[0,0] - dcmTrace)
    b_test[2] = 0.25 * (1.0 + 2*dcm[1,1] - dcmTrace)
    b_test[3] = 0.25 * (1.0 + 2*dcm[2,2] - dcmTrace)

    maxInd = np.argmax(b_test)

    quat[maxInd] = sqrt(b_test[maxInd])

    if maxInd == 0:
        quat[1] = (dcm[1,2] - dcm[2,1])/(4.0 * quat[maxInd])
        quat[2] = (dcm[2,0] - dcm[0,2])/(4.0 * quat[maxInd])
        quat[3] = (dcm[0,1] - dcm[1,0])/(4.0 * quat[maxInd])
    elif maxInd == 1:
        quat[0] = (dcm[1, 2] - dcm[2, 1]) / (4.0 * quat[maxInd])
        quat[2] = (dcm[0,1] + dcm[1,0]) / (4.0 * quat[maxInd])
        quat[3] = (dcm[2,0] + dcm[0,2])/(4.0 * quat[maxInd])
    elif maxInd == 2:
        quat[0] = (dcm[2, 0] - dcm[0, 2]) / (4.0 * quat[maxInd])
        quat[1] = (dcm[0, 1] + dcm[1, 0]) / (4.0 * quat[maxInd])
        quat[3] = (dcm[1,2] + dcm[2,1]) / (4.0 * quat[maxInd])
    elif maxInd == 3:
        quat[0] = (dcm[0, 1] - dcm[1, 0]) / (4.0 * quat[maxInd])
        quat[1] = (dcm[2,0] + dcm[0,2])/(4.0 * quat[maxInd])
        quat[2] = (dcm[1, 2] + dcm[2, 1]) / (4.0 * quat[maxInd])
    return quat

def quat2dcm(quat):
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    dcm = np.array([[q0**2.0 + q1**2.0 - q2**2.0 - q3**2.0, 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
              [2*(q1*q2 - q0*q3), q0**2.0 - q1**2.0 + q2**2.0 - q3**2.0, 2*(q2*q3+q0*q1)],
              [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), q0**2.0 - q1**2.0 - q2**2.0 + q3**2.0]])
    return dcm

def prv2crp(prvAngle, prvVector):
    rodParam = tan(prvAngle / 2.0) * prvVector
    return rodParam

def quat2crp(quat):
    rodParam = np.zeros([3])
    for ind in range(0,3):
        rodParam[ind] = quat[ind+1] / quat[0]
    return rodParam

def crp2dcm(crp):
    q1 = crp[0]
    q2 = crp[1]
    q3 = crp[2]
    coeff = 1 / (1 + np.inner(crp,crp))
    dcm = coeff * np.array([[1**2.0 + q1**2.0 - q2**2.0 - q3**2.0, 2*(q1*q2 + 1*q3), 2*(q1*q3 - 1*q2)],
              [2*(q1*q2 - 1*q3), 1**2.0 - q1**2.0 + q2**2.0 - q3**2.0, 2*(q2*q3+1*q1)],
              [2*(q1*q3 + 1*q2), 2*(q2*q3 - 1*q1), 1**2.0 - q1**2.0 - q2**2.0 + q3**2.0]])
    return dcm

def prv2mrp(prvAngle, prvVector):
    modRodParam = tan(prvAngle/4) * prvVector
    return modRodParam

def quat2mrp(quat):
    rodParam = np.zeros([3])
    for ind in range(0,3):
        rodParam[ind] = quat[ind+1] / (quat[0] + 1.0)
    return rodParam

def mrpShadow(mrp):
    mrpNorm = np.linalg.norm(mrp) ** 2.0
    shadowMrp = np.zeros([3])
    for ind in range(0,3):
        shadowMrp[ind] = -mrp[ind] / mrpNorm
    return shadowMrp

def mrp2dcm(mrp):
    sigsqr = np.inner(mrp,mrp)
    dcm = np.eye(3,3) + (8.0 * np.dot(skew(mrp),skew(mrp)) - 4.0 * (1.0 - sigsqr) * skew(mrp)) / (1 +sigsqr)**2.0
    return dcm

def mrp2quat(mrp):
    mrpNorm = np.inner(mrp,mrp)
    b0 = (1 - mrpNorm)/(1+mrpNorm)
    b1 = (2*mrp[0]) / (1+mrpNorm)
    b2 = (2 * mrp[1]) / (1 + mrpNorm)
    b3 = (2 * mrp[2]) / (1 + mrpNorm)
    quat = np.array([b0,b1,b2,b3])
    return quat

def dcm2mrp(dcm):
    zeta = sqrt(np.trace(dcm) + 1)
    sig = 1 / (zeta * (zeta+2)) * np.array([dcm[1,2]-dcm[2,1], dcm[2,0]-dcm[0,2],dcm[0,1]-dcm[1,0]])
    return sig

def mrpRate2bodyRate(mrp):
    mrpNorm = np.inner(mrp,mrp)
    Bmat = 0.25 * ((1 - mrpNorm) * np.identity(3) + 2 * skew(mrp[0:3]) + 2.0 * np.outer(mrp[0:3], mrp[0:3]))
    Binv = (4.0/(1 + mrpNorm)**2.0) * np.transpose(Bmat)
    return Binv

def dcm2euler321(dcm):
    q = np.zeros([3,])
    q[0] = atan2(dcm[0,1],dcm[0,0])
    q[1] = asin(-dcm[0,2])
    q[2] = atan2(dcm[1,2],dcm[2,2])

    return q[0],q[1],q[2]

def euler321Rate2Body(q):
    B = np.zeros([3,3])
    s2 = sin(q[1]);
    c2 = cos(q[1]);
    s3 = sin(q[2]);
    c3 = cos(q[2]);

    B[0,0] = 0
    B[0,1] = s3
    B[0,2] = c3
    B[1,0] = 0
    B[1,1] = c2*c3
    B[1,2] = -c2*s3
    B[2,0] = c2
    B[2,1] = s2*s3
    B[2,2] = s2*c3
    B = B/c2
    return B

def triad(bvec1, bvec2, nvec1, nvec2):
    #   Computes an attitude matrix from two sets of measured "body" vectors and their inertial counterparts using the TRIAD method. bvec1 and nvec1 correspond to the "more accurate" measurement.

    bvec1 = bvec1 / np.linalg.norm(bvec1)
    bvec2 = bvec2 / np.linalg.norm(bvec2)
    nvec1 = nvec1 / np.linalg.norm(nvec1)
    nvec2 = nvec2 / np.linalg.norm(nvec2)

    t1b = bvec1
    t2b = np.cross(bvec1, bvec2) / np.linalg.norm(np.cross(bvec1, bvec2))
    t3b = np.cross(t1b, t2b)

    t1n = nvec1
    t2n = np.cross(nvec1, nvec2) / np.linalg.norm(np.cross(nvec1, nvec2))
    t3n =  np.cross(t1n, t2n)

    bt = np.transpose(np.array([t1b,t2b,t3b]))
    nt = np.transpose(np.array([t1n, t2n, t3n]))
    bn = np.dot(bt,(np.transpose(nt)))
    return bn

def q_method(meas_mat, inert_mat, weightVec):
    numMeas = meas_mat.shape[0]
    B = np.zeros([3,3])
    for ind in range(0,numMeas):
        currBodyMeas = meas_mat[ind]
        print "body meas at",ind,":", currBodyMeas
        currInertMeas = inert_mat[ind]
        B = np.multiply(weightVec[ind],np.outer(currBodyMeas, currInertMeas)) + B

    S = B + np.transpose(B)
    sigma = np.trace(B)
    Z = np.array([B[1,2]-B[2,1], B[2,0]-B[0,2], B[0,1]-B[1,0]])
    a = np.hstack([sigma, Z])
    b = np.hstack([np.reshape(Z,[3,1]), S - sigma * np.identity(3)])
    K = np.vstack([a,b])
    [eigVals, eigVecs] = np.linalg.eig(K)
    print "Eigenvalaues:", eigVals
    q_est = eigVecs[:,np.argmax(eigVals)]
    q_est = q_est / np.linalg.norm(q_est)
    return q_est

def questRootFcn(s,rootOptions):
    fcnVal = np.linalg.det(rootOptions.K - s*np.identity(4))
    return fcnVal

class questOptions:
    def __init__(self, measMat):
        self.K = measMat

def quest(meas_mat, inert_mat, weightVec,tol):
    numMeas = meas_mat.shape[0]
    B = np.zeros([3, 3])

    for ind in range(0, numMeas):
        currBodyMeas = meas_mat[ind]
        currInertMeas = inert_mat[ind]
        B = np.multiply(weightVec[ind], np.outer(currBodyMeas, currInertMeas)) + B

    S = B + np.transpose(B)
    sigma = np.trace(B)
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    a = np.hstack([sigma, Z])
    b = np.hstack([np.reshape(Z, [3, 1]), S - sigma * np.identity(3)])
    K = np.vstack([a, b])
    options = questOptions(K)
    initGuess = np.sum(weightVec)
    guessedEigen = sci.optimize.fsolve(questRootFcn,  initGuess, args=options)

    crpEst = np.linalg.inv((guessedEigen+sigma)*np.identity(3) - S).dot(Z)
    return crpEst

def olae(meas_mat, inert_mat, weightVec):
    numMeas = meas_mat.shape[0]

    sumMat = np.array([skew(meas_mat[0]+inert_mat[0])])
    sumMat = np.reshape(sumMat, [3,3])

    diffMat = np.array([meas_mat[0]-inert_mat[0]])
    diffMat = np.reshape(diffMat, [3,1])

    weightMat = np.array([weightVec[0] * np.identity(3)])
    weightMat = np.reshape(weightMat, [3,3])

    for ind in range(1,numMeas):
        sumMat = np.vstack([sumMat, skew(meas_mat[ind]+inert_mat[ind])])
        diffMat = np.vstack([diffMat, np.reshape(meas_mat[ind]-inert_mat[ind],[3,1])])
        weightMat = sci.linalg.block_diag(weightMat,weightVec[ind] * np.identity(3))

    crp = np.linalg.inv( np.transpose(sumMat).dot(weightMat).dot(sumMat)).dot(np.transpose(sumMat)).dot(weightMat).dot(diffMat)
    crp = np.reshape(crp, [3,])

    return crp

def wahbahCost(meas_mat, inert_mat, Map):
    numMeas = meas_mat.shape[0]
    B = np.zeros([3,3])
    costFunction = 0.0
    for ind in range(0,numMeas):
        currBodyMeas = meas_mat[ind]
        #print currBodyMeas
        currInertMeas = inert_mat[ind]
        #print currInertMeas
        costFunctionInnard = currBodyMeas - np.dot(Map, currInertMeas)
        #print costFunctionInnard
        costFunction = costFunction + np.linalg.norm(costFunctionInnard)**2.0
    costFunction = 0.5 * costFunction

    return costFunction

class eulerOptions:
    def __init__(self, inertiaMat, torqueVec, duffCheck=False ):
        self.Inertia = inertiaMat
        self.TorqueVec = torqueVec
        self.duffCheck = duffCheck
        self.duffVec = np.zeros([3,])
        self.duffCalc = np.zeros([3,])

def mrpEulerianEOM(t0,y0,options):
    y_dot = np.zeros([6,])
    y_dot[3:] = np.linalg.inv(options.Inertia).dot(-skew(y0[3:]).dot(options.Inertia).dot(y0[3:])) + np.linalg.inv(options.Inertia).dot(options.TorqueVec )
    mrpNorm = np.inner(y0[0:3],y0[0:3])
    y_dot[0:3] = 0.25 * np.dot((1. - mrpNorm) * np.identity(3) + 2. * skew(y0[0:3]) + 2.0 * np.outer(y0[0:3],y0[0:3]), y0[3:])
    #print "Full state update:", y_dot
    if options.duffCheck == True:
        duffingConstantA, duffingConstantB = duffingComp(options.Inertia, y0[3:])
        #print "Duffing constant A:", duffingConstantA
        #print "Duffing Constant B:", duffingConstantB
        for ind in range(0,3):
            options.duffCalc[ind] = y_dot[ind+3]**2.0 + duffingConstantA[ind] * y0[ind+3]**2.0 + duffingConstantB[ind]/2.0 * y0[ind+3]**4.0
        options.duffVec = checkDuffingInt(options.Inertia, y_dot[0:3])
    return y_dot

def mrpPDcontroller(attState,gainVec):
    if type(gainVec[1]) is float:
        torqueOut = -gainVec[0] * attState[0:3] - gainVec[1] * attState[3:]
    else:
        torqueOut = -gainVec[0] * attState[0:3] - gainVec[1].dot(attState[3:])
    #print "Pos Torque Out:", -gainVec[0] * attState[0:3]
    #print "Vel torque out:", - gainVec[1] * attState[3:]
    #print "Total torque out:", torqueOut
    return torqueOut

def mrpPIDcontroller(attState,gainVec, dt, prevErrorState, initAngVel, inertiaTens):
    newErr = prevErrorState + (attState[0:3]) * dt
    torqueOut = -gainVec[0] *attState[0:3] - (gainVec[1]*np.identity(3) + gainVec[1]*gainVec[2]*inertiaTens).dot(attState[3:]) -\
                                               gainVec[0]*gainVec[1]*gainVec[2]*newErr + \
                                               gainVec[1] * gainVec[2] *inertiaTens.dot(initAngVel) + \
                                               skew(attState[3:]).dot(inertiaTens).dot(attState[3:])


    ##print "Dynamics terms:", skew(attState[3:]).dot(inertiaTens).dot(attState[3:])
    #print "MRP Term:", -gainVec[0]
    #print "Rate term:", -(gainVec[1] + gainVec[1].dot(gainVec[2]).dot(inertiaTens))
    #print "Integral term:", gainVec[0].dot(gainVec[1]).dot(gainVec[2]).dot(prevErrorState)

    return torqueOut, newErr

def euler321PDcontroller(attState, gainMats):
    angle1, angle2, angle3 = dcm2euler321( mrp2dcm(attState[0:3]))
    angleVec = np.array([angle1, angle2, angle3])
    Bmat = euler321Rate2Body(angleVec)

    torqueOut = -gainMats[1].dot(attState[3:]) - gainMats[0].dot(np.transpose(Bmat)).dot(gainMats[0]).dot(angleVec)
    return torqueOut

def quatPDcontroller(attState, gainMats):
    quat = mrp2quat(attState[0:3])
    err = quat[1:]
    torqueOut = -gainMats[0] * err - gainMats[1] * attState[3:]
    return torqueOut

def nadirPoint(scPos, mu):
    nadirDir = scPos[0:3] / np.linalg.norm(scPos[0:3])
    scMomentumDir = kl.orbitMomentumDir(scPos)
    nadirRate = kl.meanMotionComp(np.linalg.norm(scPos[0:3]), mu) * scMomentumDir
    return nadirDir, nadirRate

def targetPoint(targPos, scPos, mu):
    targDir = (targPos[0:3] - scPos[0:3]) / (np.linalg.norm(targPos[0:3]-scPos[0:3]))
    #targMomentumDir = kl.orbitMomentumDir(targPos)
    #print "Target Momentum Direction:", targMomentumDir
    #scMomentumDir = kl.orbitMomentumDir(scPos)
    #print "Spacecraft Momentum Direction:", scMomentumDir
    #targRate = kl.meanMotionComp(np.linalg.norm(targPos[0:3]), mu) * targMomentumDir - kl.meanMotionComp(np.linalg.norm(scPos),mu) * scMomentumDir
    return targDir#, targRate

def sunPoint(scPos):
    sunDir = np.array([0, 1, 0])
    sunRate = np.array([0, 0, 0])
    return sunDir, sunRate

def nVec2Att(nVec, scTransState, bodyAxis):
    pointVec = nVec / np.linalg.norm(nVec)
    orbInertiaDir = kl.orbitMomentumDir(scTransState)
    vec2 = np.cross(pointVec, orbInertiaDir) / np.linalg.norm(np.cross(pointVec, orbInertiaDir))#orbInertiaDir - np.inner(orbInertiaDir, pointVec) / np.linalg.norm(pointVec)**2.0 * pointVec
    vec3 = np.cross(pointVec, vec2) / np.linalg.norm(np.cross(pointVec, vec2))

    #print "1st and 2nd inner prod:", np.inner(pointVec, vec2)
    #print "2nd and 3rd inner prod:", np.inner(vec2, vec3)
    #print "3rd and 1st inner prod:", np.inner(vec3, pointVec)
    dcm = np.zeros([3,3])
    dcm[bodyAxis - 1, :] = pointVec
    if bodyAxis == 1:
        dcm[1,:] = vec2
        dcm[2,:] = vec3
    elif bodyAxis == 2:
        dcm[0,:] = vec3
        dcm[2,:] = vec2
    elif bodyAxis == 3:
        dcm[0,:] = vec2
        dcm[1,:] = vec3
    #print "Targeter DCM det:", np.linalg.det(dcm)
    return dcm

def duffingComp(inertia, initSpin):
    check1 = np.dot(initSpin, inertia)
    t = 0.5 * np.inner(check1, initSpin)
    h = np.linalg.norm(np.dot(inertia, initSpin))
    #print inertia
    #print initSpin
    #print "Momentum:", h
    #print "Energy:", t

    i1 = inertia[0,0]
    i2 = inertia[1,1]
    i3 = inertia[2,2]
    inertProd = i1*i2*i3

    A1 = ((i1-i2)*(2*i3*t-h**2.0)+(i1-i3)*(2*i2*t-h**2.0))/inertProd
    B1 = (2 * (i1-i2)*(i1-i3))/(i2*i3)

    A2 = ((i2 - i3) * (2 * i1 * t - h ** 2.0) + (i2 - i1)*(2 * i3 * t - h ** 2.0)) / inertProd
    B2 = (2 * (i2 - i1) * (i2 - i3)) / (i1 * i3)

    A3 = ((i3 - i1) * (2 * i2 * t - h ** 2.0) + (i3 - i2)*(2 * i1 * t - h ** 2.0)) / inertProd
    B3 = (2 * (i3 - i2) * (i3 - i1)) / (i2 * i1)

    A = np.array([A1, A2, A3])
    B = np.array([B1, B2, B3])

    return A, B

def checkDuffingInt(inertia, spin):
    t = 0.5 * sum([inertia[0,0]*spin[0]**2.0, inertia[1,1]*spin[1]**2.0, inertia[2,2]*spin[2]**2.0])
    h = np.linalg.norm(np.dot(inertia, spin))

    i1 = inertia[0, 0]
    i2 = inertia[1, 1]
    i3 = inertia[2, 2]

    k1 = (2*i2*t-h**2.0)*(h**2.0 - 2*i3*t) / (i1**2.0 * i2 * i3)
    k2 = (2*i3*t-h**2.0)*(h**2.0 - 2*i1*t) / (i1 * i2**2.0 * i3)
    k3 = (2*i1*t-h**2.0)*(h**2.0 - 2*i2*t) / (i1 * i2 * i3**2.0)

    k = np.array([k1,k2,k3])
    return k

class linearMrpOptions:
    def __init__(self, inertiaMat, mrpGain, rateGain ):
        self.Inertia = inertiaMat
        self.invInertia = np.linalg.inv(inertiaMat)
        self.mrpGain = mrpGain
        self.rateGain = rateGain

def linearMrpEOM(t0, y0, options):
    ydot = np.zeros([6,])
    ydot[0:3] = 0.25 * np.identity(3).dot(y0[3:])
    ydot[3:] = -options.mrpGain*options.invInertia.dot(y0[0:3]) - options.invInertia.dot(options.rateGain).dot(y0[3:])
    return ydot
#
# def attVisualizer(attHist):
#     bodyVec1 = np.array([1,0,0])
#     bodyVec2 = np.array([0,1,0])
#     bodyVec3 = np.array([0,0,1])
#     lineData = np.empty((dims, length))
#
#
#     def update_lines(num, dataLines, lines):
#         for line, data in zip(lines, dataLines):
#             # NOTE: there is no .set_data() for 3 dim data...
#             line.set_data(data[0:2, :num])
#             line.set_3d_properties(data[2, :num])
#         return lines
#
#     fig = plt.figure()
#     ax = p3.Axes3D(fig)
#     def update_attitude()
#     line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(attHist, lines),
#                                        interval=50, blit=False)
#

#def attTargeter(bodyAxis, inertialVector, bodyMRP, vectorMRP):
#
#
#
 #   return errorMRP

