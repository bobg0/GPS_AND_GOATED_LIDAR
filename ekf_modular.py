#Modular EKF leader‑follower tracker for AirSim

import os, sys, time, math, argparse, itertools, warnings
from collections import deque, defaultdict
sys.path.append(r"F:/AirSim/PythonClient")          # adjust if needed

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import airsim
# experiment config
class ExpConf:
    """Bundle all knobs for one experiment run."""
    def __init__(self,
                 name:str,
                 runtime:float      = 60.0,
                 loop_hz:int        = 10,
                 lidar_hz:float     = 10.0,
                 noise_model:str    = "none",
                 noise_param:float  = 0.0,
                 log_int:float      = 1.0):
        self.name        = name
        self.runtime     = runtime
        self.loop_dt     = 1.0/loop_hz
        self.lidar_dt    = 1.0/lidar_hz
        self.noise_model = noise_model   # none|gaussian|student|cauchy
        self.noise_param = noise_param
        self.log_int     = log_int
# preset shortcuts
PRESETS = {
    "baseline":           ExpConf("baseline"),
    "temporal_sparsity":  ExpConf("temporal_sparsity", lidar_hz=0.2, log_int=5),
}

# helper function: noise injection to the provided vector given default setting
def inject_noise(vec:np.ndarray, model:str, param:float) -> np.ndarray:
    if model=="none":     return vec
    if model=="gaussian": return vec + np.random.randn(*vec.shape)*param
    if model=="student":  return vec + np.random.standard_t(df=2,size=vec.shape)*param
    if model=="cauchy":   return vec + np.random.standard_cauchy(size=vec.shape)*param
    return vec

# identical leader trajectory code 
class LeaderTrajectory:
    def __init__(self,width=40.0,height=10.0,base_alt=60.0,
                 alt_amplitude=5.0,period=60.0):
        self.A,self.B = width/2.0,height/2.0
        self.base_alt = base_alt
        self.alt_amp  = alt_amplitude
        self.period   = period
        self.omega    = 2.0*math.pi/period
        self.f_z      = 0.05
    def get_position(self,t:float):
        x = self.A*math.sin(self.omega*t)
        y = self.B*math.sin(2*self.omega*t)
        z = -(self.base_alt + self.alt_amp*math.sin(2*math.pi*self.f_z*t))
        return (x,y,z)

class LidarTracker:
    def __init__(self): self.last_centroid=None
    def update(self,lidar_data,drone_position):
        pts=np.asarray(lidar_data.point_cloud,dtype=np.float32)
        if pts.size<3: return None,0
        pts=pts.reshape(-1,3)
        q=[lidar_data.pose.orientation.x_val,
           lidar_data.pose.orientation.y_val,
           lidar_data.pose.orientation.z_val,
           lidar_data.pose.orientation.w_val]
        R_mat=R.from_quat(q).as_matrix()
        pts_world=(R_mat@pts.T).T+np.array([
            lidar_data.pose.position.x_val,
            lidar_data.pose.position.y_val,
            lidar_data.pose.position.z_val],dtype=np.float32)
        mask=np.abs(pts_world[:,2]-drone_position.z_val)<=10.0
        pts_world=pts_world[mask]
        if pts_world.shape[0]==0: return None,0
        labels=DBSCAN(eps=0.5,min_samples=2).fit_predict(pts_world)
        uniq=set(labels)-{-1}
        if not uniq: return None,0
        centroids={l:pts_world[labels==l].mean(0) for l in uniq}
        counts   ={l:(labels==l).sum()            for l in uniq}
        if self.last_centroid is None:
            best=max(counts,key=counts.get)
        else:
            best=min(centroids,
                     key=lambda l:np.linalg.norm(centroids[l]-self.last_centroid))
        self.last_centroid=centroids[best]
        return tuple(self.last_centroid),counts[best]

class FollowerChaser:
    def __init__(self,client,speed=5.0):
        self.client,self.speed=client,speed
    def chase(self,tgt,name="Drone1"):
        if tgt is None: return
        st=self.client.getMultirotorState(vehicle_name=name)
        gps, pos=st.gps_location, st.kinematics_estimated.position
        Re=6371000.0
        dN,dE,dD=tgt[0]-pos.x_val, tgt[1]-pos.y_val, tgt[2]-pos.z_val
        tgt_lat=gps.latitude +(dN/Re)*180/math.pi
        tgt_lon=gps.longitude+(dE/(Re*math.cos(math.radians(gps.latitude))))*180/math.pi
        tgt_alt=gps.altitude - dD
        self.client.moveToGPSAsync(tgt_lat,tgt_lon,tgt_alt,
                                   self.speed,vehicle_name=name)
        
# drawing the true position in red, draw estimated position in green 
class Visualizer:
    def __init__(self,client): self.client=client
    def draw(self,est,true):
        if est is not None:
            self.client.simPlotPoints([airsim.Vector3r(*est)],[0,1,0,1],10,0.3,False)
        if true is not None:
            self.client.simPlotPoints([airsim.Vector3r(*true)],[1,0,0,1],10,0.3,False)

# constant acceleration motion model with jerk
class ConstantAccelEKF:
    def __init__(self,jerk_sigma=0.5):
        self.x=np.zeros((9,1)); self.P=np.eye(9)*10.0; self.jerk_sigma=jerk_sigma
    def F(self,dt):
        F=np.eye(9)
        F[0,3]=F[1,4]=F[2,5]=dt
        F[0,6]=F[1,7]=F[2,8]=0.5*dt**2
        F[3,6]=F[4,7]=F[5,8]=dt
        return F
    def Q(self,dt):
        q=self.jerk_sigma**2
        G=np.zeros((9,3))
        G[0:3,:]=0.5*dt**2*np.eye(3)
        G[3:6,:]=dt*np.eye(3)
        G[6:9,:]=np.eye(3)
        return G@(q*np.eye(3))@G.T
    def predict(self,dt):
        F=self.F(dt)
        self.x=F@self.x
        self.P=F@self.P@F.T+self.Q(dt)
    def update(self,z,H,R):
        y=z.reshape(-1,1)-H@self.x
        S=H@self.P@H.T+R
        K=self.P@H.T@np.linalg.inv(S)
        self.x=self.x+K@y
        self.P=(np.eye(9)-K@H)@self.P
    @property
    def pos(self): return self.x[0:3,0]

# wrapper function for running variations of experiments
def run_experiment(cfg:ExpConf,client:airsim.MultirotorClient):
    traj    = LeaderTrajectory(base_alt=60.0)
    tracker = LidarTracker()
    chaser  = FollowerChaser(client)
    viz     = Visualizer(client)
    ekf     = ConstantAccelEKF()

    for n in ("Drone1","Drone2"):
        client.enableApiControl(True,n)
        client.armDisarm(True,n)
        client.takeoffAsync(vehicle_name=n).join()
        client.moveToZAsync(-50,3,vehicle_name=n).join()

    st0=client.getMultirotorState("Drone2").kinematics_estimated.position
    ekf.x[0:3,0]=np.array([st0.x_val,st0.y_val,st0.z_val])

    log=defaultdict(list)
    start=time.perf_counter()
    last_loop=start
    last_lidar=0.0
    last_log  =0.0

    # control loop with logical time
    while True:
        now=time.perf_counter()
        sim_t=now-start
        if sim_t>cfg.runtime: break
        dt=now-last_loop; last_loop=now

        p_rel     = traj.get_position(sim_t)
        p_next_rel= traj.get_position(sim_t+dt)
        v_cmd     = [(p_next_rel[i]-p_rel[i])/dt*2 for i in range(3)]
        self_v    = [float(v) for v in v_cmd]
        client.moveByVelocityAsync(self_v[0],self_v[1],self_v[2],
                                   duration=float(dt),vehicle_name="Drone2")

        ekf.predict(dt)

        if sim_t-last_lidar >= cfg.lidar_dt:
            lidar=client.getLidarData("LidarSensor2","Drone1")
            follower_pos=client.getMultirotorState("Drone1").\
                          kinematics_estimated.position
            meas,_=tracker.update(lidar,follower_pos)
            # noise injection
            if meas is not None:
                z=inject_noise(np.array(meas),cfg.noise_model,cfg.noise_param)
                H=np.hstack([np.eye(3),np.zeros((3,6))])
                R=np.eye(3)*0.1**2
                ekf.update(z,H,R)
            last_lidar=sim_t
        # drone1 update position based on estimation
        chaser.chase(tuple(ekf.pos))
        viz.draw(tuple(ekf.pos),
                 tuple(client.getMultirotorState("Drone2").kinematics_estimated.
                       position.__iter__()))
        # logging update
        if sim_t-last_log >= cfg.log_int:
            p_true=client.getMultirotorState("Drone2").kinematics_estimated.position
            p_true=np.array((p_true.x_val,p_true.y_val,p_true.z_val))
            err=np.linalg.norm(p_true-ekf.pos)
            log["t"].append(sim_t)
            log["true"].append(p_true)
            log["est"].append(ekf.pos.copy())
            log["err"].append(err)
            log["cov"].append(np.diag(ekf.P)[0:3])
            last_log=sim_t

        lag=time.perf_counter()-now
        if lag<cfg.loop_dt: time.sleep(cfg.loop_dt-lag)

    for n in ("Drone1","Drone2"):
        client.armDisarm(False,n); client.enableApiControl(False,n)
    return log

# plot relevant stats & est vs. gt
def save_plots(log,outdir,label):
    t=np.array(log["t"])
    true=np.vstack(log["true"])
    est =np.vstack(log["est"])
    err=np.array(log["err"])
    cov=np.vstack(log["cov"])
    os.makedirs(outdir,exist_ok=True)
    for i,ax in enumerate("XYZ"):
        plt.figure()
        plt.plot(t,true[:,i],label=f"LD {ax}")
        plt.plot(t,est [:,i],label=f"FD {ax}")
        plt.xlabel("Time [s]"); plt.ylabel(f"{ax}[m]"); plt.legend()
        plt.title(label); plt.savefig(os.path.join(outdir,f"{label}_traj_{ax}.png")); plt.close()
    plt.figure()
    plt.plot(t,err,label="‖x̂−x‖")
    σ=np.sqrt(cov.sum(1))
    plt.fill_between(t,3*σ,-3*σ,color="grey",alpha=0.3,label="±3σ")
    plt.xlabel("Time [s]"); plt.ylabel("error [m]"); plt.legend(); plt.title(label)
    plt.tight_layout(); plt.savefig(os.path.join(outdir,f"{label}_err.png")); plt.close()
    plt.figure()
    for i,ax in enumerate("xyz"):
        plt.plot(t,cov[:,i],label=f"P{ax}{ax}")
    plt.yscale("log"); plt.xlabel("Time[s]"); plt.legend(); plt.title(label)
    plt.savefig(os.path.join(outdir,f"{label}_cov.png")); plt.close()

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--exp",choices=["baseline","temporal_sparsity",
                                   "lidar_rate_sweep","noise_robustness"],
                   required=True)
    p.add_argument("--runtime",type=float)
    p.add_argument("--log-int",type=float)
    p.add_argument("--lidar-rates",nargs="+",type=float,
                   help="list when exp=lidar_rate_sweep")
    p.add_argument("--noise",choices=["gaussian","student","cauchy"])
    p.add_argument("--sigma","--scale",dest="noise_param",type=float)
    return p.parse_args()

def main():
    args=parse()
    client=airsim.MultirotorClient(); client.confirmConnection()
    cfgs=[]
    if args.exp=="lidar_rate_sweep":
        if not args.lidar_rates:
            raise SystemExit("--lidar-rates required for lidar_rate_sweep")
        for r in args.lidar_rates:
            cfgs.append(ExpConf(f"lidar_{r}Hz",lidar_hz=r,
                                runtime=args.runtime or 60,
                                log_int=args.log_int or 1))
    elif args.exp=="noise_robustness":
        model=args.noise or "gaussian"
        param=args.noise_param if args.noise_param is not None else 3.0
        cfgs.append(ExpConf(f"noise_{model}",noise_model=model,
                             noise_param=param,runtime=args.runtime or 60,
                             log_int=args.log_int or 1))
    else:
        base=PRESETS[args.exp]
        if args.runtime : base.runtime = args.runtime
        if args.log_int : base.log_int = args.log_int
        cfgs.append(base)

    for cfg in cfgs:
        print(f"==> {cfg.name}")
        log=run_experiment(cfg,client)
        outdir=os.path.join(os.getcwd(),"ekf_plots",cfg.name)
        save_plots(log,outdir,cfg.name)
        print(f"   plots saved → {outdir}")

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=UserWarning, module="matplotlib")
    main()
