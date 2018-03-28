import numpy as np
import math
import keplerLib as kl
import scipy as sci
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import schaubLib as sl

def solarSysJ2(planet):
    if planet == "Moon":
        return 0.0002027
    elif planet == "Mercury":
        return 0.00006
    elif planet == "Venus":
        return 0.000027
    elif planet == "Earth":
        return 0.0010826269
    elif planet == "Mars":
        return 0.001964
    elif planet == "Jupiter":
        return 0.01475
    elif planet == "Saturn":
        return 0.01645
    elif planet == "Uranus":
        return 0.012
    elif planet == "Neptune":
        return 0.004
    else:
        return 1e9

def omegaDotCalc(meanMotion,planetRadius,J2Val, semiLatus, incl):
    top = -3.0 * meanMotion * (planetRadius**2.0) * J2Val * math.cos(incl)
    bot = 2.0 * semiLatus**2.0
    #print "top", top
    #print "bot", bot
    return top / bot

class odeOptions:
    def __init__(self, mu, acc, j2):
        self.mu = mu
        self.j2 = j2
        self.acc = acc

def propModel(t,y,odeOptions):
    mu = odeOptions.mu
    acc = odeOptions.acc
    #print y
    radius = np.linalg.norm(y[0:3])
    #print radius
    gravAcc = -mu/(radius**3.0)
    ctStm = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[gravAcc,0,0,0,0,0],[0,gravAcc,0,0,0,0],[0,0,gravAcc,0,0,0]])
    ctStm = np.reshape(ctStm, [6,6])
    #print ctStm
    addAcc = np.array([[0],[0],[0],[acc[0]],[acc[1]],[acc[2]]])
    addAcc = np.reshape(addAcc, [6,])
    y_dot = np.dot(ctStm, y) + addAcc
    #print y_dot
    #y_dot = np.vstack([y_dot, (-mu / np.linalg.norm(y[0:3])**3.0)*y[0:3]+acc])
    y_dot = np.reshape(y_dot, [6,])
    #print y_dot[3:6]
    return y_dot

def j2PropModel(t,y,odeOptions):
    mu = odeOptions.mu
    acc = odeOptions.acc
    j2 = solarSysJ2("Earth")
    planetRad = kl.planetRadius("Earth")
    radius = np.linalg.norm(y[0:3])

    gravDir = y[0:3] / radius
    x_eci = y[0]
    y_eci = y[1]
    z_eci = y[2]

    y_dot = np.zeros([6,])
    #j2_coeff = -3./2. * (mu*j2*planetRad**2.0)/(radius**4.0)
    #j2_vec = np.array([(1. - 3.*z_eci**2.0/radius**2.0)*x_eci/radius,
    #                   (1. - 3. * z_eci ** 2.0 / radius ** 2.0) * y_eci / radius,
    #                   (3. - 3. * z_eci ** 2.0 / radius ** 2.0) * x_eci / radius])
    a = np.zeros([3,])
    a[0] = - 3*j2*mu*planetRad**2*y[0]/(2*radius**5)*(1.-5.*y[2]**2/radius**2)
    a[1] = - 3*j2*mu*planetRad**2*y[1]/(2*radius**5)*(1.-5.*y[2]**2/radius**2)
    a[2] = - 3*j2*mu*planetRad**2*y[2]/(2*radius**5)*(3.-5.*y[2]**2/radius**2)
    y_dot[0:3] = y[3:]
    y_dot[3:] = -mu/(radius**2.0) * gravDir + a + acc

    return y_dot


def perturbedProp(pos, vel, acc, mu,time):
    init_vec = np.vstack([pos,vel])
    init_vec = np.reshape(init_vec, [6,])
    #print init_vec
    intOptions = odeOptions(mu,acc,solarSysJ2("Earth"))
    final_vec = sl.rk4(propModel,time[0],time[1]-time[0],init_vec, intOptions)
    final_pos = final_vec[0:3]
    final_vel = final_vec[3:]
    return final_pos, final_vel

def j2PerturbedProp(pos, vel, acc, mu, time):
    init_vec = np.vstack([pos,vel])
    init_vec = np.reshape(init_vec, [6,])
    intOptions = odeOptions(mu,acc,solarSysJ2("Earth"))
    final_vec = sl.rk4(j2PropModel,time[0],time[1]-time[0],init_vec, intOptions)

    final_pos = final_vec[0:3]
    final_vel = final_vec[3:]
    return final_pos, final_vel

def perturbationDeltas(semi,radius,ecc,meanMot,incl,raan, peri, nu,dt,forces):
    #   Force components in RSW frame
    Fr = forces[0]
    Fs = forces[1]
    Fw = forces[2]

    #   Calculation of other necessary variables
    latus = semi * (1-ecc**2.0)
    argLat = nu+peri
    #print "meanMot:", meanMot
    #rint "Semi:", semi
    #print "Ecc:", ecc
    angMom = meanMot * semi**2.0 * math.sqrt(1.0-ecc**2.0)
    #   Implementation of equation 9-24 in Vallado
    dsemi_dt = 2/(meanMot * math.sqrt(1-ecc**2.0)) * (ecc*math.sin(nu)*Fr + latus/radius * Fs )

    decc_dt = math.sqrt(1-ecc**2.0) / (meanMot*semi) * (math.sin(nu) * Fr + (math.cos(nu) + (ecc+math.cos(nu))/(1+ecc*math.cos(nu)))*Fs)

    dincl_dt = (radius*math.cos(argLat) * Fw) / (meanMot*semi**2.0 * math.sqrt(1-ecc**2.0))

    draan_dt = radius*math.sin(argLat) * Fw / (meanMot * semi**2.0 * math.sin(incl) * math.sqrt(1-ecc**2.0))

    dperi_dt = math.sqrt(1 - ecc**2.0) / (meanMot * semi * ecc) * (-math.cos(nu) * Fr + math.sin(nu) * (1 - radius/latus) * Fs) - (radius * (1/math.tan(incl) * math.sin(argLat)) * Fw) / angMom

    #dmean_dt = (1 / (meanMot * semi**2.0 * ecc))* ((latus * math.cos(nu) - 2*ecc*radius) * Fr - (latus+radius) * math.sin(nu) * Fs) -
    dnu_dt = (angMom / radius**2.0) + (1 / ecc*angMom) * (latus*math.cos(nu))*Fr - (latus+radius) * math.sin(nu) * Fs
    return dsemi_dt, decc_dt, dincl_dt, draan_dt, dperi_dt, dnu_dt

def dragCalc(dragCoeff, atmoDens, relVel, projArea):
    dragForce = 0.5 * dragCoeff * atmoDens * projArea * np.linalg.norm(relVel)**2.0 * (-relVel / np.linalg.norm(relVel))
    return dragForce

def liftCalc(liftCoeff, atmoDens, relVel, projArea):
    liftForce = 0.5 * liftCoeff * atmoDens * projArea * np.linalg.norm(relVel) ** 2.0 * (relVel / np.linalg.norm(relVel))
    return liftForce

def expAtmo(baseDensity, scaleHeight, pos):
    alt = np.linalg.norm(pos) - kl.planetRadius("Earth")
    atmoDens = baseDensity * math.exp(-alt/scaleHeight)
    return atmoDens

def earthExpAtmo(pos):
    alt = np.linalg.norm(pos) - kl.planetRadius("Earth")
    baseDensity = 1.020 #   kg/m^3
    scaleHeight = 8.0e3 #   In meters
    atmoDens = baseDensity * math.exp(-alt/scaleHeight)
    return atmoDens

def simpleDrag(ballisticCoeff, atmoDens, relVel):
    dragForce = -0.5 * atmoDens * ballisticCoeff * np.linalg.norm(relVel) * relVel
    return dragForce


def marsAtmoProps(pos):
    alt = np.linalg.norm(pos) - kl.planetRadius("Mars")
    #temp = -23.4 - 0.00222 * alt
    temp = 150.0
    pressure = 0.699 * math.exp(-0.00009 * (alt-6000))
    dens = 0.020 * math.exp(-alt / 11.1e3)
    #dens = 1e-13
    marsDay = 24.6597 * 3600
    marsOmega = (360.0 * math.pi/180.0)/marsDay
    omegaVec = np.array([[0],[0],[marsOmega]])
    surfVel = marsOmega * kl.planetRadius("Mars")
    rotAngMom = surfVel * kl.planetRadius("Mars")
    relVel = rotAngMom / np.linalg.norm(pos)
    #print relVel
    return temp, pressure, dens, relVel

class flatPlate:
    def __init__(self, plateAreas, plateDragCoeff, plateLiftCoeff, plateNormals, mass):
        self.plateAreas = plateAreas
        self.plateNormals = plateNormals
        self.plateDragCoeff = plateDragCoeff
        self.plateLiftCoeff = plateLiftCoeff
        self.mass = mass
        self.numPlates = len(plateAreas)

def flatPlateAero(flatPlate, bhAtt, H_eciVel):
    H_eciVelhat = H_eciVel / np.linalg.norm(H_eciVel)
    dragAreaSum = 0.0

    if flatPlate.numPlates == 1:

        rotNormal = sl.mrp2dcm(bhAtt).dot(flatPlate.plateNormals[:])
        velProj = np.inner(rotNormal,H_eciVelhat)

        #   If the plate normal is opposed to the velocity vector...
        if velProj < 0.0:
            #   Compute drag forces
            dragAreaSum = dragAreaSum + flatPlate.plateDragCoeff * flatPlate.plateAreas * -velProj

    else:
        for ind in range(0,flatPlate.numPlates):

            rotNormal = sl.mrp2dcm(bhAtt).dot(flatPlate.plateNormals[ind])
            velProj = np.inner(rotNormal, H_eciVelhat)

            #   If the plate normal is opposed to the velocity vector...
            if velProj < 0.0:
                #   Compute drag forces
                dragAreaSum = dragAreaSum + flatPlate.plateDragCoeff[ind] * flatPlate.plateAreas[ind] * -velProj

    balCoeff = dragAreaSum / flatPlate.mass
    #print balCoeff
    return balCoeff

def linearDragController(relativeState):
    gainMat = np.loadtxt('linDragGain.csv',delimiter=',')
    desAtt = -gainMat.dot(relativeState)
    if np.linalg.norm(desAtt) > 1.0:
        desAtt = sl.mrpShadow(desAtt)
    return desAtt

def gsiPlateAero(fullArea,  bankAngles, relVel, accomCoeff, scTemp, atmoDens, atmoMass, atmoTemp, flag):
    #plateNormal = np.array([[0.0], [1.0], [0.0]])
    relVel = -relVel
    gasConst = 8.3144598 #/ atmoMass
    incVel = np.linalg.norm(relVel)
    incTemp = incVel**2.0 / (3*gasConst/atmoMass)


    remVel = incVel * math.sqrt((2.0/3.0) * (1.0 + accomCoeff * (scTemp/incTemp - 1.0)))

    #otNormal = np.dot(kl.rot3(bankAngles[0])*kl.rot2(bankAngles[1])*kl.rot1(bankAngles[2]),plateNormal)
    rotNormal = -relVel / incVel
    rotNormal = np.dot(np.dot(np.dot(kl.rot1(bankAngles[0]), kl.rot2(bankAngles[1])), kl.rot3(bankAngles[2])), rotNormal)
    velDir = relVel / incVel
    dragDir= np.dot(np.transpose(rotNormal),velDir)



    VcrossN = np.cross(velDir,rotNormal,axis=0)

    tripleCross= np.cross(VcrossN,velDir,axis=0)
    liftDir = np.dot(np.transpose(tripleCross),rotNormal)

    liftDir = np.reshape(liftDir,[1])
    dragDir = np.reshape(dragDir, [1])
    projArea = abs(dragDir) * fullArea

    speedRat = incVel / math.sqrt((2.0*gasConst*atmoTemp/atmoMass))

    P = (math.exp(-(dragDir**2.0 * speedRat**2.0)) ) / speedRat
    Q = 1.0 + 1.0/(2.0*speedRat**2.0)
    G = 1.0/(2.0*speedRat**2.0)
    Z = 1.0 + math.erf(-dragDir*speedRat)

    Cd =  -((P / math.sqrt(math.pi)) + dragDir * Q * Z + (dragDir * remVel)/(2*incVel) * (dragDir * math.sqrt(math.pi) * Z + P))
    Cl =  (liftDir * G * Z + (liftDir * remVel)/(2*incVel) * (dragDir * math.sqrt(math.pi) * Z + P))

    #Cd = 2.0
    #Cl = 100.0

    dragForce =  0.5 * Cd * projArea * atmoDens * np.linalg.norm(relVel)**2.0
    liftForce =  0.5 * Cl * projArea * atmoDens * np.linalg.norm(relVel)**2.0

    dragVec = dragForce * velDir
    liftVec = liftForce * tripleCross

    dragVec = np.reshape(dragVec, [3,])
    liftVec = np.reshape(liftVec, [3,])
    if flag == 1:
        print "Rotated normal:", rotNormal
        print "Velocity Vector:", velDir
        print "Lift Vector:", tripleCross
        print "error function:", math.erf(dragDir * speedRat)
        print "Remitted Velocity:", remVel
        print "Incoming Velocity:", incVel
        print "Rotated normal vector x:", rotNormal[0]
        print "Rotated normal vector y:", rotNormal[1]
        print "Rotated normal vector z:", rotNormal[2]
        print "X Vel: ", relVel[0]
        print "Y Vel: ", relVel[1]
        print "Z Vel: ", relVel[2]
        print "Incoming Temp:", incTemp
        print "speed ratio:", speedRat
        print "Dot between normal and drag vec:", dragDir
        print "Dot between normal and lift vec:", liftDir
        print "Drag force, N:", dragForce
        print "Lift force, N:", liftForce
        print "Drag Vector:", dragVec
        print "Lift Vector:", liftVec
        print "Projected area:", projArea
        print "P: ", P
        print "Q: ", Q
        print "G: ", G
        print "Z: ", Z

    return Cd, Cl, dragVec, liftVec, dragDir, projArea


def mrpGsiPlateAero(plateGeometry,  plateMrp, relVel, accomCoeff, scTemp, atmoDens, atmoMass, atmoTemp, flag):

    relVel = -relVel
    gasConst = 8.3144598 #/ atmoMass
    incVel = np.linalg.norm(relVel)
    incTemp = incVel**2.0 / (3*gasConst/atmoMass)


    remVel = incVel * math.sqrt((2.0/3.0) * (1.0 + accomCoeff * (scTemp/incTemp - 1.0)))

    #   Compute velocity projection
    gamma = (np.inner(plateGeometry.plateNormals[:], -np.array([0,1,0])))

    rotNormal = -relVel / incVel
    rotNormal = sl.mrp2dcm(plateMrp).dot(rotNormal)
    velDir = relVel / incVel
    dragDir= np.dot(np.transpose(rotNormal),velDir)



    VcrossN = np.cross(velDir,rotNormal,axis=0)

    tripleCross= np.cross(VcrossN,velDir,axis=0)
    liftDir = np.dot(np.transpose(tripleCross),rotNormal)

    liftDir = np.reshape(liftDir,[1])
    dragDir = np.reshape(dragDir, [1])
    projArea = abs(dragDir) * fullArea

    speedRat = incVel / math.sqrt((2.0*gasConst*atmoTemp/atmoMass))

    P = (math.exp(-(dragDir**2.0 * speedRat**2.0)) ) / speedRat
    Q = 1.0 + 1.0/(2.0*speedRat**2.0)
    G = 1.0/(2.0*speedRat**2.0)
    Z = 1.0 + math.erf(-dragDir*speedRat)

    Cd =  -((P / math.sqrt(math.pi)) + dragDir * Q * Z + (dragDir * remVel)/(2*incVel) * (dragDir * math.sqrt(math.pi) * Z + P))
    Cl =  (liftDir * G * Z + (liftDir * remVel)/(2*incVel) * (dragDir * math.sqrt(math.pi) * Z + P))

    #Cd = 2.0
    #Cl = 100.0

    dragForce =  0.5 * Cd * projArea * atmoDens * np.linalg.norm(relVel)**2.0
    liftForce =  0.5 * Cl * projArea * atmoDens * np.linalg.norm(relVel)**2.0

    dragVec = dragForce * velDir
    liftVec = liftForce * tripleCross

    dragVec = np.reshape(dragVec, [3,])
    liftVec = np.reshape(liftVec, [3,])

    return Cd, Cl, dragVec, liftVec, dragDir, projArea