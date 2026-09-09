#!/usr/bin/env python3
"""Replay logged poses with the actual slab and target, without replay dynamics."""
import argparse,os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','glfw');os.environ.setdefault('DISPLAY',':1')
import numpy as np,mujoco,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
ap=argparse.ArgumentParser()
for k in ['model','qpos','metrics','out','title']:ap.add_argument('--'+k,required=True)
ap.add_argument('--face',type=float,required=True);ap.add_argument('--target',required=True);ap.add_argument('--croco',action='store_true');ap.add_argument('--keep_table_legs',action='store_true');ap.add_argument('--fps',type=int,default=20)
a=ap.parse_args();target=np.array([float(v) for v in a.target.split(',')])
m=mujoco.MjModel.from_xml_path(str(Path(a.model).resolve()));d=mujoco.MjData(m)
tb=m.body('table').id;tg=m.geom('table_top_collision').id;dz=a.face-(m.body_pos[tb,2]+m.geom_pos[tg,2]+m.geom_size[tg,2]);m.body_pos[tb,2]+=dz
under=m.geom_pos[tg,2]-m.geom_size[tg,2]
for name in ['table_leg_'+str(i)+suffix for i in range(1,5) for suffix in ['','_collision']]:
 i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name)
 if i>=0 and not a.keep_table_legs:m.geom_size[i,2]=max(.01,.5*(under+m.body_pos[tb,2]));m.geom_pos[i,2]=under-m.geom_size[i,2]
x=np.genfromtxt(a.qpos,names=True,delimiter=',',skip_header=3 if a.croco else 0)
t=x['time' if a.croco else 't'];q=np.column_stack([x[('qpos' if a.croco else 'q')+str(i)] for i in range(m.nq)])
s=np.genfromtxt(a.metrics,names=True,delimiter=',')
good=np.where((s['reach_error']<=.07)&(s['brace_N']>=20)&(s['trunk_N']<10))[0]
still_t=float(s['t'][good[len(good)//2]]) if len(good) else t[-1]-2
cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.azimuth=145;cam.elevation=-12;cam.distance=3.05;cam.lookat[:]=[.65,-.03,1.]
old_target=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'target')
if old_target>=0:
 m.geom_rgba[m.geom_bodyid==old_target,3]=0 # orange sphere is this run's actual goal
m.vis.global_.offwidth=800;m.vis.global_.offheight=608
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
writer=imageio_ffmpeg.write_frames(str(out),(800,608),fps=a.fps,quality=7,macro_block_size=16,output_params=['-threads','1']);writer.send(None)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
with mujoco.Renderer(m,height=608,width=800) as ren:
 for k,tt in enumerate(np.arange(t[0],t[-1],1/a.fps)):
  i=min(np.searchsorted(t,tt),len(t)-1);d.qpos[:]=q[i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
  ren.update_scene(d,camera=cam)
  scene=ren.scene
  if scene.ngeom<scene.maxgeom:
   mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.018,0,0]),target,np.eye(3).ravel(),np.array([1.,.25,.05,.9],dtype=np.float32));scene.ngeom+=1
  img=Image.fromarray(ren.render());dr=ImageDraw.Draw(img);idx=min(np.searchsorted(s['t'],tt),len(s)-1);r=s[idx]
  text=[a.title,'Face %.3f m | t %.2f s | orange: commanded target'%(a.face,tt),'Reach %.0f mm | left-arm %.0f N | trunk %.0f N'%(1000*r['reach_error'],r['brace_N'],r['trunk_N'])]
  dr.rectangle((0,0,800,78),fill=(20,25,32))
  for j,line in enumerate(text):dr.text((12,4+24*j),line,font=font,fill='white')
  writer.send(np.asarray(img))
  if abs(tt-still_t)<.5/a.fps:img.save(out.with_suffix('.png'))
writer.close();print(out)
