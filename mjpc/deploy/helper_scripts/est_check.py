import time, numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
ChannelFactoryInitialize(0, "enx00e04c681314")
E=[]; A=[]; Y=[]
def ce(m): E.append((m.position[0],m.position[1],m.position[2]))
def ca(m):
    if int(m.mode)==2: A.append((m.position[0],m.position[1])); Y.append(m.position[2])
s1=ChannelSubscriber("rt/sportmodestate", SportModeState_); s1.Init(ce,10)
s2=ChannelSubscriber("rt/aux_odom", SportModeState_); s2.Init(ca,10)
t0=time.time()
while time.time()-t0<3.0: time.sleep(0.05)
e=np.median(np.array(E),0) if E else None
a=np.median(np.array(A),0) if A else None
yaw=float(np.median(Y)) if Y else 0.0
# est publishes the IMU SITE (pelvis + R*IMU_OFFSET); anchor is the PELVIS -> remove the lever arm (yaw only)
OFF=np.array([-0.04452,-0.01891]); c,s_=np.cos(yaw),np.sin(yaw)
ro=np.array([c*OFF[0]-s_*OFF[1], s_*OFF[0]+c*OFF[1]])
if e is not None: e=np.array([e[0]-ro[0], e[1]-ro[1], e[2]])
gap=None if (e is None or a is None) else np.hypot(e[0]-a[0],e[1]-a[1])
print("EST pelvis xy=(%s)  |  ABS anchor xy=(%s)  |  gap=%s  %s"%(
    None if e is None else "%.3f,%.3f z=%.3f"%tuple(e),
    None if a is None else "%.3f,%.3f"%tuple(a),
    None if gap is None else "%.1fcm"%(gap*100),
    "CONVERGED" if (gap is not None and gap<0.03) else "not converged / mismatch"))
# 2026-08-29 placement check: reach into B3 needs the pelvis <= ~32 cm from the near edge
# (29_37 grasped from 24 cm; 29_39/40 stalled 10-15 cm short from 41/29 cm).
if a is not None:
    dist_cm=(0.45-a[0])*100.0
    print("PLACEMENT: pelvis %.0f cm from the near table edge, %.0f cm %s of centreline  -> %s"%(
        dist_cm, abs(a[1])*100.0, "LEFT" if a[1]>0 else "RIGHT",
        "OK" if dist_cm<=32 else "MOVE ROBOT FORWARD ~%.0f cm (want <=30)"%(dist_cm-30)))

