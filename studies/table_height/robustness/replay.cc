// Independent evaluator linked to EXACTLY the bench's MuJoCo revision.
// No MJPC task, planner, sensor callback, or controller state is involved.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
int main(int argc,char**argv){
 if(argc!=5){std::fprintf(stderr,"usage: replay model.mjb state.csv metrics.csv model_info.json\n");return 2;}
 mjModel*m=mj_loadModel(argv[1],nullptr);if(!m)return 2;mjData*d=mj_makeData(m);
 auto body=[&](const char*n){return mj_name2id(m,mjOBJ_BODY,n);};
 int tb=body("table"),pelvis=body("pelvis"),torso=body("torso_link"),grip=body("right_magpie_gripper");
 int tg=mj_name2id(m,mjOBJ_GEOM,"table_top_collision");
 if(std::min({tb,pelvis,torso,grip,tg})<0)return 2;
 std::vector<int>robot(m->nbody),left(m->nbody);for(int b=0;b<m->nbody;b++){
  int parent=b;while(parent>0&&parent!=pelvis)parent=m->body_parentid[parent];robot[b]=parent==pelvis;
  std::string n=mj_id2name(m,mjOBJ_BODY,b)?mj_id2name(m,mjOBJ_BODY,b):"";
  left[b]=robot[b]&&n.rfind("left_",0)==0&&(n.find("shoulder")!=std::string::npos||n.find("elbow")!=std::string::npos||n.find("wrist")!=std::string::npos||n.find("gripper")!=std::string::npos||n.find("magpie")!=std::string::npos);
 }
 double face=m->body_pos[3*tb+2]+m->geom_pos[3*tg+2]+m->geom_size[3*tg+2];
 double near=m->body_pos[3*tb]+m->geom_pos[3*tg]-m->geom_size[3*tg];
 int feet[2]={body("left_ankle_roll_link"),body("right_ankle_roll_link")};double feet0[4]={};
 FILE*info=std::fopen(argv[4],"w");std::fprintf(info,"{\"version\":\"%s\",\"nq\":%d,\"nv\":%d,\"nu\":%d,\"face\":%.17g,\"near\":%.17g,\"center\":%.17g,\"physics_dt\":%.17g}\n",mj_versionString(),m->nq,m->nv,m->nu,face,near,m->body_pos[3*tb+1]+m->geom_pos[3*tg+1],m->opt.timestep);std::fclose(info);
 std::ifstream in(argv[2]);std::string line;std::getline(in,line);
 FILE*out=std::fopen(argv[3],"w");if(!out)return 2;
 std::fprintf(out,"t,phase,jaw_x,jaw_y,jaw_z,brace_N,brace_up_N,trunk_N,other_N,pelvis_z,tilt_deg,joint_velocity,joint_limit_rad,actuator_force_fraction,foot_displacement,penetration_m,ctrl_violation\n");
 int row=0;while(std::getline(in,line)){
  std::stringstream stream(line);std::string field;std::vector<double>x;while(std::getline(stream,field,','))x.push_back(std::stod(field));
  if(x.size()!=size_t(2+m->nq+2*m->nv+m->nu)){std::fprintf(stderr,"bad row\n");return 2;}
  int k=0;d->time=x[k++];int phase=int(x[k++]);for(int j=0;j<m->nq;j++)d->qpos[j]=x[k++];for(int j=0;j<m->nv;j++)d->qvel[j]=x[k++];for(int j=0;j<m->nu;j++)d->ctrl[j]=x[k++];for(int j=0;j<m->nv;j++)d->qacc_warmstart[j]=x[k++];
  mj_forward(m,d);
  mjtNum local[3]={.2254,-.0118,-.1062},jaw[3];mju_mulMatVec3(jaw,d->xmat+9*grip,local);mju_addTo3(jaw,d->xpos+3*grip);
  double bn=0,bu=0,tn=0,on=0,penetration=0;for(int c=0;c<d->ncon;c++){
   const mjContact&co=d->contact[c];int a=m->geom_bodyid[co.geom[0]],b=m->geom_bodyid[co.geom[1]];
   if(robot[a]||robot[b])penetration=std::max(penetration,-co.dist);
   if((a==tb)==(b==tb))continue;int other=a==tb?b:a;if(!robot[other])continue;
   mjtNum f[6];mj_contactForce(m,d,c,f);double fn=std::max(0.,f[0]);
   if(left[other]){bn+=fn;bu+=std::max(0.,(other==b?1.:-1.)*co.frame[2]*fn);}else if(other==pelvis||other==torso)tn+=fn;else on+=fn;
  }
  double vel=0,jlim=0,frac=0,slip=0,ctrl=0;for(int j=0;j<m->njnt;j++)if(robot[m->jnt_bodyid[j]]&&m->jnt_type[j]==mjJNT_HINGE){
   vel=std::max(vel,std::abs(d->qvel[m->jnt_dofadr[j]]));if(m->jnt_limited[j]){double q=d->qpos[m->jnt_qposadr[j]];jlim=std::max({jlim,m->jnt_range[2*j]-q,q-m->jnt_range[2*j+1]});}}
  for(int j=0;j<m->nu;j++){if(m->actuator_forcelimited[j]){double den=std::max(std::abs(m->actuator_forcerange[2*j]),std::abs(m->actuator_forcerange[2*j+1]));if(den>0)frac=std::max(frac,std::abs(d->actuator_force[j])/den);}ctrl=std::max({ctrl,m->actuator_ctrlrange[2*j]-d->ctrl[j],d->ctrl[j]-m->actuator_ctrlrange[2*j+1]});}
  for(int j=0;j<2;j++){if(row==0){feet0[2*j]=d->xpos[3*feet[j]];feet0[2*j+1]=d->xpos[3*feet[j]+1];}slip=std::max(slip,std::hypot(d->xpos[3*feet[j]]-feet0[2*j],d->xpos[3*feet[j]+1]-feet0[2*j+1]));}
  double tilt=std::acos(std::clamp(d->xmat[9*torso+8],-1.,1.))*180./3.141592653589793;
  std::fprintf(out,"%.9g,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",d->time,phase,jaw[0],jaw[1],jaw[2],bn,bu,tn,on,d->qpos[2],tilt,vel,jlim,frac,slip,penetration,ctrl);row++;
 }
 std::fclose(out);mj_deleteData(d);mj_deleteModel(m);return 0;
}
