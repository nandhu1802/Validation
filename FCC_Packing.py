# =============================================================================
# FCC Triaxial Test - Cui & O'Sullivan (2005)
# Quarter cylinder - circumferential periodic boundaries + stress membrane
#
# Target peak stress ratios (Table 1, periodic boundary rows):
#   f=0.05    -> ~2.09
#   f=0.12278 -> ~2.46  (default, 3.9% below Thornton theory)
#   f=0.2     -> ~2.92
#   f=0.3     -> ~3.64
#
# FIXES applied over uploaded version:
#   1. Clip radius: cylRadius - r  ->  cylRadius + r  (outermost layer was excluded)
#   2. O.periodic removed: YADE periodic cell makes ALL 6 faces periodic, paper
#      only requires x=0 and y=0. Replaced with explicit Python image particles.
#   3. Membrane command restored (was missing entirely - no sigma3 confinement)
#   4. Faceted rigid cylindrical outer wall removed (not in paper)
#   5. specArea: cylR**2 -> (cylR+r)**2  (outer particle surface, not centres)
#   6. Orphaned '...' and duplicate TAGS section removed
# =============================================================================

from yade import utils, export, O
from yade.pack import *
from math import sqrt, atan2, pi
import os, glob, math, sys
import numpy as np
from scipy.spatial import Voronoi

# =============================================================================
# PARAMETERS  (Cui & O'Sullivan 2005, Section 4.2)
# =============================================================================
r             = 0.02
density       = 2000.0
kn            = 1.5e9
ks            = 1.5e9
# YADE Ip2_FrictMat_FrictMat_FrictPhys contact stiffness for equal spheres:
#   kn_contact = young * r_eff,  r_eff = r1*r2/(r1+r2) = r/2
# To get kn_contact = kn: young = kn / (r/2) = 2*kn/r
young         = 2.0 * kn / r  # 1.5e11 Pa -> kn_contact = young*(r/2) = kn
poissonRatio  = ks / kn       # 1.0       -> ks_contact = kn_contact*1.0 = ks
frictionCoeff = 0.12278       # Rowe (1962)
confStress    = 5.0e4         # sigma3 [Pa]
loadVel       = 0.01          # [m/s] per platen
damping       = 0.3
targetStrain  = 0.025
nLayers       = 16
nRings        = 8

outCsv = "fcc_triaxial_results.csv"
vtkDir = "vtk_fcc"
ckpDir = "checkpoints_fcc"
for d in [vtkDir+"/spheres", vtkDir+"/interactions", vtkDir+"/walls", ckpDir]:
    if not os.path.exists(d):
        os.makedirs(d)

# =============================================================================
# FCC PACKING - quarter cylinder
#
# a = 2r: exact touching. verletDist on the collider detects these contacts.
# Quarter-cylinder clip: x>=0, y>=0, radial dist <= cylRadius + r
#   (+ r so outermost particle layer is included - FIX 1)
# =============================================================================
a         = 2.0 * r
dy_row    = a * sqrt(3.0) / 2.0
layerZ    = a * sqrt(2.0 / 3.0)
dxB       = a / 2.0
dyB       = a * sqrt(3.0) / 6.0
cylRadius = nRings * a

all_centres = []
for k in range(nLayers):
    zPos = k * layerZ
    for j in range(2*nRings+2):
        for i in range(2*nRings+2):
            if k % 2 == 0:
                x = i*a + (j%2)*0.5*a
                y = j*dy_row
            else:
                x = i*a + (j%2)*0.5*a + dxB
                y = j*dy_row + dyB
            x -= cylRadius
            y -= cylRadius
            all_centres.append((x, y, zPos))

# FIX 1: was cylRadius - r (excluded outermost layer), now cylRadius + r
centres = [c for c in all_centres
           if c[0] >= 0.0 and c[1] >= 0.0
           and sqrt(c[0]**2 + c[1]**2) <= cylRadius + r]

if not centres:
    raise RuntimeError("No particles generated.")

print("Particles: %d" % len(centres))
zBot = min(c[2] for c in centres)
zTop = max(c[2] for c in centres)

# =============================================================================
# MATERIAL
# =============================================================================
fccMat = O.materials.append(FrictMat(
    young        = young,
    poisson      = poissonRatio,
    density      = density,
    frictionAngle= 0.0,
    label        = 'fccMat'
))

wallMat = O.materials.append(FrictMat(
    young        = young,
    poisson      = poissonRatio,
    density      = density,
    frictionAngle= 0.0,   # frictionless walls
    label        = 'wallMat'
))

# =============================================================================
# SPHERES
# =============================================================================
for c in centres:
    O.bodies.append(sphere(c, r, material='fccMat'))

sphereIds = [b.id for b in O.bodies if isinstance(b.shape, Sphere)]
print("Sphere bodies: %d" % len(sphereIds))

# =============================================================================
# MEMBRANE PARTICLES AND CYLINDER SURFACE RADIUS  (FIX 3 - was missing)
#
# Paper Sec 3.2: R = mean radial distance of membrane particle centres
# specArea = pi*(R+r)^2/4  (outer surface, not just centres - FIX 5)
# =============================================================================
rdists = {b.id: sqrt(b.state.pos[0]**2 + b.state.pos[1]**2)
          for b in O.bodies if isinstance(b.shape, Sphere)}
maxRad = max(rdists.values())
membraneIds = [bid for bid, rd in rdists.items() if abs(rd - maxRad) <= 2.0*r]
if not membraneIds:
    raise RuntimeError("No membrane particles found. maxRad=%.4f" % maxRad)
cylR     = sum(rdists[bid] for bid in membraneIds) / len(membraneIds)
# FIX 5: was cylR**2, now (cylR+r)**2
specArea = 0.25 * pi * (cylR + r)**2
print("Membrane particles: %d  cylR=%.4f m  area=%.4e m^2"
      % (len(membraneIds), cylR, specArea))

# =============================================================================
# PLATENS
# YADE box half-extent in z = r
#   bottom inner face at zBot-r  =>  centre at zBot-2r
#   top    inner face at zTop+r  =>  centre at zTop+2r
# =============================================================================
platHalf    = cylRadius + 2.0*r
platBotCenZ = zBot - 2.0*r
platTopCenZ = zTop + 2.0*r

bottomWall = O.bodies.append(box(
    center =(0.0, 0.0, platBotCenZ),
    extents=(platHalf, platHalf, r),
    fixed=True, material='wallMat'))
O.bodies[bottomWall].label = 'bottomWall'

topWall = O.bodies.append(box(
    center =(0.0, 0.0, platTopCenZ),
    extents=(platHalf, platHalf, r),
    fixed=True, material='wallMat'))
O.bodies[topWall].label = 'topWall'

botFaceZ    = platBotCenZ + r
topFaceZ    = platTopCenZ - r
initPlatSep = topFaceZ - botFaceZ
print("Platen separation: %.6f m" % initPlatSep)
print("Bottom inner face z = %.6f  (zBot-r = %.6f)" % (botFaceZ, zBot-r))
print("Top    inner face z = %.6f  (zTop+r = %.6f)" % (topFaceZ, zTop+r))

# =============================================================================
# CIRCUMFERENTIAL PERIODIC BOUNDARIES  (FIX 2 - replaced O.periodic)
#
# Paper Sec 3.1, Fig 2: virtual-ball approach at x=0 and y=0 cut-planes.
# For each real particle within r of x=0: image at (-x, y, z)
# For each real particle within r of y=0: image at (x, -y, z)
# F-type particles (on z-axis): no images, blocked in xy.
#
# O.periodic=True was WRONG: it makes all 6 box faces periodic, causing
# spurious contacts across the top/bottom platens and outer curved surface.
# This manual image-particle approach matches the paper exactly.
#
# Engine order each step:
#   1. updateImages()      - mirror positions BEFORE contact detection
#   2. InsertionSortCollider
#   3. InteractionLoop     - computes contact forces incl. image contacts
#   4. membraneCmd         - applies radial membrane forces
#   5. routeForces()       - routes image forces to real parents BEFORE Newton
#   6. NewtonIntegrator    - integrates real particles only (images blocked)
# =============================================================================
axisTol = 0.5 * r
axisIds = set()
for b in O.bodies:
    if isinstance(b.shape, Sphere):
        if sqrt(b.state.pos[0]**2 + b.state.pos[1]**2) <= axisTol:
            b.state.blockedDOFs = 'xyXY'
            axisIds.add(b.id)
print("F-type (axis) particles: %d" % len(axisIds))

imageMap = {}  # {realId: [('x', imgId), ('y', imgId)]}
planeEps = 1e-10  # threshold: particle is ON the cut plane (not just near it)
nOnX = 0; nOnY = 0
for b in list(O.bodies):
    if not isinstance(b.shape, Sphere): continue
    if b.id in axisIds: continue
    bx, by, bz = b.state.pos
    imgs = []
    if bx < planeEps:
        # Particle sits exactly on x=0 plane; image would overlap self → explosion.
        # Enforce symmetry by blocking the x translational and rotational DOFs.
        b.state.blockedDOFs = b.state.blockedDOFs + 'xX' if b.state.blockedDOFs else 'xX'
        nOnX += 1
    elif bx <= r:
        iid = O.bodies.append(sphere((-bx, by, bz), r, material='fccMat'))
        O.bodies[iid].state.blockedDOFs = 'xyzXYZ'
        imgs.append(('x', iid))
    if by < planeEps:
        # Particle sits exactly on y=0 plane; block y DOFs instead of creating image.
        existing = b.state.blockedDOFs
        if 'y' not in existing:
            b.state.blockedDOFs = existing + 'yY'
        nOnY += 1
    elif by <= r:
        iid = O.bodies.append(sphere((bx, -by, bz), r, material='fccMat'))
        O.bodies[iid].state.blockedDOFs = 'xyzXYZ'
        imgs.append(('y', iid))
    if imgs:
        imageMap[b.id] = imgs
print("On x=0 plane (blocked, no image): %d" % nOnX)
print("On y=0 plane (blocked, no image): %d" % nOnY)

nImgs = sum(len(v) for v in imageMap.values())
print("Periodic image particles: %d" % nImgs)

def updateImages():
    """Mirror real particle kinematics to image particles.
    Must run BEFORE InsertionSortCollider."""
    for realId, imgList in imageMap.items():
        br = O.bodies[realId]
        rx, ry, rz = br.state.pos
        vx, vy, vz = br.state.vel
        wx, wy, wz = br.state.angVel
        for axis, iid in imgList:
            bi = O.bodies[iid]
            if axis == 'x':
                bi.state.pos    = (-rx,  ry,  rz)
                bi.state.vel    = Vector3(-vx,  vy,  vz)
                bi.state.angVel = Vector3( wx, -wy, -wz)  # pseudovector: flip wy,wz under x-reflection
            else:
                bi.state.pos    = ( rx, -ry,  rz)
                bi.state.vel    = Vector3( vx, -vy,  vz)
                bi.state.angVel = Vector3(-wx,  wy, -wz)  # pseudovector: flip wx,wz under y-reflection

def routeForces():
    """Transfer contact forces from image particles to real parents with
    correct sign inversion. Must run AFTER InteractionLoop, BEFORE Newton.
    Reads from i.phys.normalForce/shearForce (not O.forces which is cleared
    by ForceResetter at start of each step)."""
    imgToReal = {}
    for realId, imgList in imageMap.items():
        for axis, iid in imgList:
            imgToReal[iid] = (realId, axis)
    for i in O.interactions:
        if not i.isReal or not hasattr(i, 'phys') or i.phys is None: continue
        nf = i.phys.normalForce
        sf = i.phys.shearForce
        # YADE convention: normalForce = force ON id2.
        # Image is id2: force on image = +(nf+sf); reflect x-component to route to real.
        if i.id2 in imgToReal:
            realId, axis = imgToReal[i.id2]
            fx = nf[0]+sf[0]; fy = nf[1]+sf[1]; fz = nf[2]+sf[2]
            if axis == 'x': O.forces.addF(realId, Vector3(-fx,  fy,  fz))
            else:           O.forces.addF(realId, Vector3( fx, -fy,  fz))
        # Image is id1: force on image = -(nf+sf); reflect x-component to route to real.
        if i.id1 in imgToReal:
            realId, axis = imgToReal[i.id1]
            fx = -(nf[0]+sf[0]); fy = -(nf[1]+sf[1]); fz = -(nf[2]+sf[2])
            if axis == 'x': O.forces.addF(realId, Vector3(-fx,  fy,  fz))
            else:           O.forces.addF(realId, Vector3( fx, -fy,  fz))

# =============================================================================
# TAGS
# =============================================================================
theoreticalRatio = 2.0*(1.0+frictionCoeff)/(1.0-frictionCoeff)

O.tags['topWallId']        = str(topWall)
O.tags['bottomWallId']     = str(bottomWall)
O.tags['specArea']         = str(specArea)
O.tags['sigma3']           = str(confStress)
O.tags['r']                = str(r)
O.tags['initPlatSep']      = str(initPlatSep)
O.tags['outCsv']           = outCsv
O.tags['memIds']           = " ".join(map(str, membraneIds))
O.tags['cylR']             = str(cylR)
O.tags['phase']            = "consolidation"
O.tags['lastStrain']       = "0.0"
O.tags['targetStrain']     = str(targetStrain)
O.tags['theoreticalRatio'] = "%.4f" % theoreticalRatio
O.tags['ckpDir']           = ckpDir

with open(outCsv, 'w') as f:
    f.write("iter,simTime,phase,axialStrain,sigma1_kPa,sigma3_kPa,stressRatio\n")

vtkS = export.VTKExporter(vtkDir+"/spheres/spheres")
vtkI = export.VTKExporter(vtkDir+"/interactions/interactions")
vtkW = export.VTKExporter(vtkDir+"/walls/walls")
vtkS.exportSpheres(what={'membrane':'1 if b.id in %r else 0' % set(membraneIds)})
print("[VTK] Initial snapshot saved.")

# =============================================================================
# MEMBRANE COMMAND  (FIX 3 - was entirely missing)
#
# Paper Sec 3.2, Eqs 1-2:
#   Unroll cylinder surface S onto plane S':
#     x' = theta * R   (arc-length)
#     z' = z
#   Voronoi tessellation on S' gives tributary area A_i per membrane particle.
#   Radial inward force: F_i = sigma3 * A_i
#
# 8-direction mirrors (3x3 minus self) bound all Voronoi cells incl. corners.
# Applied every step (iterPeriod=1) - continuous confinement as per paper.
# =============================================================================
membraneCmd = (
    "from math import sqrt, atan2\n"
    "from scipy.spatial import Voronoi\n"
    "import numpy as np\n"
    "sig3 = float(O.tags['sigma3'])\n"
    "R    = float(O.tags['cylR'])\n"
    "mIds = [int(x) for x in O.tags['memIds'].split()]\n"
    "proj = []; pdat = []\n"
    "for bid in mIds:\n"
    "    b = O.bodies[bid]\n"
    "    px,py,pz = b.state.pos\n"
    "    rr = sqrt(px*px+py*py)\n"
    "    th = atan2(py,px)\n"
    "    proj.append([th*R, pz])\n"
    "    pdat.append((bid,px,py,rr))\n"
    "if len(proj) >= 4:\n"
    "    pts = np.array(proj)\n"
    "    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):\n"
    "        import sys; print('MEMBRANE: NaN/Inf in positions, skipping step'); sys.stdout.flush()\n"
    "    else:\n"
    "        xs  = max(pts[:,0].max()-pts[:,0].min(), 1e-12)\n"
    "        zs  = max(pts[:,1].max()-pts[:,1].min(), 1e-12)\n"
    "        mir = []\n"
    "        for p in proj:\n"
    "            for dx in [-xs,0,xs]:\n"
    "                for dz in [-zs,0,zs]:\n"
    "                    if dx!=0 or dz!=0:\n"
    "                        mir.append([p[0]+dx, p[1]+dz])\n"
    "        vor = Voronoi(np.vstack([pts, np.array(mir)]))\n"
    "        def area(v,i):\n"
    "            reg = v.regions[v.point_region[i]]\n"
    "            if -1 in reg or len(reg)<3: return 0.0\n"
    "            vx=v.vertices[reg]; n=len(vx); s=0.0\n"
    "            for ii in range(n):\n"
    "                jj=(ii+1)%n\n"
    "                s+=vx[ii,0]*vx[jj,1]-vx[jj,0]*vx[ii,1]\n"
    "            return abs(s)*0.5\n"
    "        for idx,(bid,dx,dy,rr) in enumerate(pdat):\n"
    "            a = area(vor,idx)\n"
    "            if a>0 and rr>0:\n"
    "                F = sig3*a\n"
    "                O.forces.addF(bid, Vector3(-F*dx/rr,-F*dy/rr,0.0))\n"
)

# =============================================================================
# LOGGER
# Forces read from i.phys.normalForce - not O.forces (reset each step)
# =============================================================================
loggerCmd = (
    "topId  = int(O.tags['topWallId'])\n"
    "botId  = int(O.tags['bottomWallId'])\n"
    "A      = float(O.tags['specArea'])\n"
    "sig3   = float(O.tags['sigma3'])\n"
    "phase  = O.tags['phase']\n"
    "tStr   = float(O.tags['targetStrain'])\n"
    "rp     = float(O.tags['r'])\n"
    "L0     = float(O.tags['initPlatSep'])\n"
    "topZ   = O.bodies[topId].state.pos[2] - rp\n"
    "botZ   = O.bodies[botId].state.pos[2] + rp\n"
    "L      = topZ - botZ\n"
    "eps    = max((L0-L)/L0, 0.0) if L0>0 else 0.0\n"
    "fTop   = sum((i.phys.normalForce[2] if i.id1==topId else -i.phys.normalForce[2])\n"
    "             for i in O.interactions\n"
    "             if i.isReal and hasattr(i,'phys') and i.phys is not None\n"
    "             and (i.id1==topId or i.id2==topId))\n"
    "fBot   = sum((i.phys.normalForce[2] if i.id1==botId else -i.phys.normalForce[2])\n"
    "             for i in O.interactions\n"
    "             if i.isReal and hasattr(i,'phys') and i.phys is not None\n"
    "             and (i.id1==botId or i.id2==botId))\n"
    "sig1   = 0.5*(fTop-fBot)/A if A>0 else 0.0\n"
    "ratio  = sig1/(-sig3) if sig3>0 else 0.0\n"
    "with open(O.tags['outCsv'],'a') as f:\n"
    "    f.write('%d,%.6f,%s,%.6f,%.4f,%.4f,%.4f\\n'%(\n"
    "        O.iter,O.time,phase,eps,sig1/1e3,-sig3/1e3,ratio))\n"
    "lastEps = float(O.tags['lastStrain'])\n"
    "if eps-lastEps >= 0.001:\n"
    "    O.tags['lastStrain'] = str(eps)\n"
    "    import sys\n"
    "    print('STRAIN [%s] iter=%d eps=%.4f sig1=%.2fkPa ratio=%.3f'%(\n"
    "        phase.upper(),O.iter,eps,sig1/1e3,ratio))\n"
    "    sys.stdout.flush()\n"
    "if phase=='shearing' and eps>=tStr:\n"
    "    import sys\n"
    "    print('STOP eps=%.4f ratio=%.4f theory=%s'%(\n"
    "        eps,ratio,O.tags['theoreticalRatio']))\n"
    "    sys.stdout.flush()\n"
    "    O.pause()\n"
)

# =============================================================================
# PROGRESS  (every 1000 iters)
# Monitor with:  tail -f yade.*.out
# =============================================================================
progressCmd = (
    "topId  = int(O.tags['topWallId'])\n"
    "botId  = int(O.tags['bottomWallId'])\n"
    "A      = float(O.tags['specArea'])\n"
    "sig3   = float(O.tags['sigma3'])\n"
    "phase  = O.tags['phase']\n"
    "rp     = float(O.tags['r'])\n"
    "L0     = float(O.tags['initPlatSep'])\n"
    "topZ   = O.bodies[topId].state.pos[2] - rp\n"
    "botZ   = O.bodies[botId].state.pos[2] + rp\n"
    "L      = topZ - botZ\n"
    "eps    = max((L0-L)/L0, 0.0) if L0>0 else 0.0\n"
    "fTop   = sum((i.phys.normalForce[2] if i.id1==topId else -i.phys.normalForce[2])\n"
    "             for i in O.interactions\n"
    "             if i.isReal and hasattr(i,'phys') and i.phys is not None\n"
    "             and (i.id1==topId or i.id2==topId))\n"
    "fBot   = sum((i.phys.normalForce[2] if i.id1==botId else -i.phys.normalForce[2])\n"
    "             for i in O.interactions\n"
    "             if i.isReal and hasattr(i,'phys') and i.phys is not None\n"
    "             and (i.id1==botId or i.id2==botId))\n"
    "sig1   = 0.5*(fTop-fBot)/A if A>0 else 0.0\n"
    "ratio  = sig1/(-sig3) if sig3>0 else 0.0\n"
    "nC     = sum(1 for i in O.interactions if i.isReal)\n"
    "uMean  = sum(b.state.vel.norm() for b in O.bodies\n"
    "             if isinstance(b.shape,Sphere))/max(len(O.bodies),1)\n"
    "import sys\n"
    "print('PROGRESS [%s] iter=%d t=%.4fs eps=%.5f '\n"
    "      'sig1=%.2fkPa sig3=%.2fkPa ratio=%.4f contacts=%d meanVel=%.3e'%(\n"
    "    phase.upper(),O.iter,O.time,eps,\n"
    "    sig1/1e3,-sig3/1e3,ratio,nC,uMean))\n"
    "sys.stdout.flush()\n"
)

# =============================================================================
# VTK AND CHECKPOINTS
# =============================================================================
vtkCmd = (
    "try:\n"
    "    vtkS.exportSpheres(what={'vel':'b.state.vel.norm()'})\n"
    "    vtkI.exportInteractions(what={'fn':'i.phys.normalForce.norm()'})\n"
    "    import sys; print('VTK iter=%d'%O.iter); sys.stdout.flush()\n"
    "except Exception as e:\n"
    "    print('VTK error:',e)\n"
)

ckpCmd = (
    "import os,glob,sys\n"
    "d=O.tags['ckpDir']; ph=O.tags['phase']\n"
    "fn='%s/ckp_%s_%08d.yade.gz'%(d,ph,O.iter)\n"
    "O.save(fn)\n"
    "print('CKP saved:',fn); sys.stdout.flush()\n"
    "existing=sorted(glob.glob('%s/ckp_%s_*.yade.gz'%(d,ph)))\n"
    "for old in existing[:-3]: os.remove(old)\n"
)

# =============================================================================
# ENGINES
# Order is critical:
#   1. ForceResetter
#   2. updateImages()         - sync image positions BEFORE contact detection
#   3. InsertionSortCollider  - detect contacts (real + image)
#   4. InteractionLoop        - compute contact forces
#   5. membraneCmd            - apply radial membrane forces (every step)
#   6. routeForces()          - route image forces to real parents BEFORE Newton
#   7. NewtonIntegrator       - integrate real particles only
#   8. loggerCmd, progressCmd, vtkCmd, ckpCmd
# =============================================================================
O.engines = [
    ForceResetter(),
    PyRunner(iterPeriod=1,     command='updateImages()'),
    InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Box_Aabb()],
                          verletDist=0.05*r),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom(), Ig2_Box_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_FrictPhys()],
        [Law2_ScGeom_FrictPhys_CundallStrack()]
    ),
    PyRunner(iterPeriod=1,     command=membraneCmd),
    PyRunner(iterPeriod=1,     command='routeForces()'),
    NewtonIntegrator(damping=damping, gravity=(0,0,0), label='newton'),
    PyRunner(iterPeriod=200,   command=loggerCmd),
    PyRunner(iterPeriod=1000,  command=progressCmd),
    PyRunner(iterPeriod=10000, command=vtkCmd),
    PyRunner(iterPeriod=50000, command=ckpCmd),
]

# =============================================================================
# TIMESTEP
# =============================================================================
O.dt = utils.PWaveTimeStep()
print("dt = %.4e s" % O.dt)

O.step()
nC = len(O.interactions)
print("Contacts after init: %d" % nC)
if nC == 0:
    raise RuntimeError("Zero contacts at init.")

# Check for already-moving particles after first step
fast = [(b.id, b.state.vel.norm(), b.state.pos)
        for b in O.bodies if isinstance(b.shape,Sphere)
        and b.state.vel.norm() > 1.0]
if fast:
    bid, v, pos = fast[0]
    print("WARNING: fast particle at iter=1: id=%d vel=%.2f pos=%s" % (bid,v,pos))

print("Theory ratio (Thornton 1979): %.4f" % theoreticalRatio)
print("Expected with periodic BC   : ~2.46 (paper Table 1, f=0.12278)")

# =============================================================================
# CONSOLIDATION
# Paper Sec 4.2: sigma3 applied, cycled to equilibrium, friction=0.
# Convergence: meanVel < 1e-4 m/s or 200k iters.
# =============================================================================
print("\n" + "="*60)
print("PHASE 1: CONSOLIDATION  sigma3=%.1f kPa" % (confStress/1e3))
print("Monitor:  tail -f yade.*.out")
print("="*60)

for block in range(40):
    O.run(5000, True)
    uMean = sum(b.state.vel.norm() for b in O.bodies
                if isinstance(b.shape,Sphere)) / max(len(sphereIds),1)
    nC = sum(1 for i in O.interactions if i.isReal)
    print("CONSOL block=%d iter=%d meanVel=%.3e contacts=%d"
          % (block+1, O.iter, uMean, nC))
    sys.stdout.flush()
    if uMean < 1e-4:
        print("CONSOL equilibrium reached.")
        sys.stdout.flush()
        break
else:
    print("CONSOL WARNING: 200k iters reached, proceeding.")
    sys.stdout.flush()

O.save("%s/ckp_post_consolidation.yade.gz" % ckpDir)
print("CKP post-consolidation saved.")
sys.stdout.flush()

topFace_pc    = O.bodies[topWall].state.pos[2] - r
botFace_pc    = O.bodies[bottomWall].state.pos[2] + r
postConsolSep = topFace_pc - botFace_pc
O.tags['initPlatSep'] = str(postConsolSep)
print("Post-consolidation L0 = %.6f m" % postConsolSep)
sys.stdout.flush()

# =============================================================================
# SHEARING
# Paper Sec 4.2: friction activated, platens moved at 0.01 L/T each.
# =============================================================================
print("\n" + "="*60)
print("PHASE 2: SHEARING  f=%.5f  v=%.4e m/s" % (frictionCoeff, loadVel))
print("="*60)

O.materials[fccMat].frictionAngle = math.atan(frictionCoeff)
for inter in O.interactions:
    if inter.isReal and hasattr(inter,'phys') and inter.phys is not None:
        inter.phys.tangensOfFrictionAngle = frictionCoeff

O.tags['phase']      = "shearing"
O.tags['lastStrain'] = "0.0"

O.bodies[topWall].state.blockedDOFs    = 'xyzXYZ'
O.bodies[topWall].state.vel            = Vector3(0, 0, -loadVel)
O.bodies[bottomWall].state.blockedDOFs = 'xyzXYZ'
O.bodies[bottomWall].state.vel         = Vector3(0, 0,  loadVel)

print("Shearing running...  tail -f yade.*.out")
sys.stdout.flush()
O.run(wait=True)

print("\nSIMULATION COMPLETE")
print("Results : %s" % outCsv)
print("Theory  : %.4f" % theoreticalRatio)
print("Expected: ~2.46 (periodic BC, paper Table 1, f=0.12278)")
