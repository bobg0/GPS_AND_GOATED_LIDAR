import os
import sys
import math
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r"F:/AirSim/PythonClient") 
import airsim

SAVE_PATH = r"F:\AirSim\PythonClient\E205_Final_Proj\GPS_EKF_PLOTS"

EARTH_R = 6_378_137.0
RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0

# velocity to latitude conversion
def vel_mps_to_deg(lat_deg, vx, vy):
    lat_rad = lat_deg * DEG2RAD
    return vx / EARTH_R * RAD2DEG, vy / (EARTH_R * math.cos(lat_rad)) * RAD2DEG

#acceleration to radiants conversion
def acc_mps2_to_deg2(lat_deg, ax, ay):
    lat_rad = lat_deg * DEG2RAD
    return ax / EARTH_R * RAD2DEG, ay / (EARTH_R * math.cos(lat_rad)) * RAD2DEG

#leader trajectory
class LeaderEKF:
    def __init__(self, dt, Q, R_pos, R_vel):
        self.dt = dt
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1e-3
        self.Q, self.Rp, self.Rv = Q, R_pos, R_vel
        F = np.eye(6); F[0,3]=F[1,4]=F[2,5]=dt; self.F = F
        B = np.zeros((6,3)); dt2=0.5*dt*dt
        B[0,0]=B[1,1]=B[2,2]=dt2; B[3,0]=B[4,1]=B[5,2]=dt; self.B=B

    def predict(self, acc_deg):
        self.x = self.F @ self.x + self.B @ acc_deg.reshape(3,1)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_pos(self, z):
        H = np.hstack((np.eye(3), np.zeros((3,3))))
        y = z.reshape(3,1) - H @ self.x
        S = H @ self.P @ H.T + self.Rp
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6)-K@H) @ self.P

    def update_vel(self, z):
        H = np.hstack((np.zeros((3,3)), np.eye(3)))
        y = z.reshape(3,1) - H @ self.x
        S = H @ self.P @ H.T + self.Rv
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6)-K@H) @ self.P
# traking helper
def run_ekf_tracking(gps_rate=50, noise_type='gaussian', log_interval=5.0, exp_name='exp'):
    os.makedirs(SAVE_PATH, exist_ok=True)
    cli = airsim.MultirotorClient(); cli.confirmConnection()
    for d in ["Drone2","Drone1"]:
        cli.enableApiControl(True,d); cli.armDisarm(True,d)

    cli.takeoffAsync(vehicle_name="Drone2").join()
    cli.moveToZAsync(-3,1,vehicle_name="Drone2").join()
    cli.takeoffAsync(vehicle_name="Drone1").join()
    cli.moveToZAsync(-3,1,vehicle_name="Drone1").join()

    dt = 1.0 / gps_rate
    ekf = LeaderEKF(
        dt,
        Q = np.diag([1e-4,1e-4,0.02, 2e-4,2e-4,0.02]),
        R_pos = np.diag([2e-4,2e-4,0.5]),
        R_vel = np.diag([2e-4,2e-4,0.5])
    )

    gps0 = cli.getGpsData("","Drone2").gnss.geo_point
    kin0 = cli.simGetGroundTruthKinematics("Drone2")
    dlat0, dlon0 = vel_mps_to_deg(gps0.latitude, kin0.linear_velocity.x_val, kin0.linear_velocity.y_val)
    ekf.x[:,0] = np.array([gps0.latitude, gps0.longitude, gps0.altitude, dlat0, dlon0, kin0.linear_velocity.z_val])

    if noise_type == 'gaussian':
        noise_func = lambda scale: np.random.randn() * scale
    elif noise_type == 'cauchy':
        noise_func = lambda scale: np.random.standard_cauchy() * scale
    elif noise_type == 'student':
        noise_func = lambda scale: np.random.standard_t(df=3) * scale
    else:
        raise ValueError(f"Unknown noise type {noise_type}")

    gps_noise_deg = 8e-5
    gps_noise_alt = 0.5
    vel_noise_mps = 0.8
    acc_noise_mps2 = 2.0

    gps_lag_s = 0.3
    imu_lag_s = 0.1
    gps_delay = int(gps_lag_s/dt)
    imu_delay = int(imu_lag_s/dt)
    gps_buffer, imu_buffer = [], []

    A,B,omega = 10,5,0.2
    z_amp,f_z,max_speed = 3.0,0.05,2.0
    total = 60.0; t = 0.0
    true_log, est_log, time_log, P_log, error_log = [],[],[],[],[]
    next_log_time = 0.0

    while t < total:
        tic = time.time()

        vx = A*omega*math.cos(omega*t)
        vy = 2*B*omega*math.cos(2*omega*t)
        vz = z_amp*2*math.pi*f_z*math.cos(2*math.pi*f_z*t)
        vmag = math.sqrt(vx*vx + vy*vy + vz*vz)
        if vmag > max_speed:
            vx *= max_speed/vmag; vy *= max_speed/vmag; vz *= max_speed/vmag

        futL = cli.moveByVelocityAsync(vx,vy,vz,dt, drivetrain=airsim.DrivetrainType.ForwardOnly,
                                       yaw_mode=airsim.YawMode(False,0), vehicle_name="Drone2")
        # assuming true position from client side 
        gps = cli.getGpsData("","Drone2").gnss.geo_point
        lat_raw = gps.latitude + noise_func(gps_noise_deg)
        lon_raw = gps.longitude + noise_func(gps_noise_deg)
        alt_raw = gps.altitude + noise_func(gps_noise_alt)
        kin = cli.simGetGroundTruthKinematics("Drone2")

        vx_n = kin.linear_velocity.x_val + noise_func(vel_noise_mps)
        vy_n = kin.linear_velocity.y_val + noise_func(vel_noise_mps)
        vz_n = kin.linear_velocity.z_val + noise_func(vel_noise_mps)
        # obtain "raw measurements"
        dlat_raw, dlon_raw = vel_mps_to_deg(lat_raw, vx_n, vy_n)
        dalt_raw = vz_n

        ax_n = kin.linear_acceleration.x_val + noise_func(acc_noise_mps2)
        ay_n = kin.linear_acceleration.y_val + noise_func(acc_noise_mps2)
        az_n = kin.linear_acceleration.z_val + noise_func(acc_noise_mps2)
        ddlat_raw, ddlon_raw = acc_mps2_to_deg2(lat_raw, ax_n, ay_n)
        ddalt_raw = az_n

        gps_buffer.append(np.array([lat_raw, lon_raw, alt_raw]))
        imu_buffer.append((np.array([dlat_raw, dlon_raw, dalt_raw]), np.array([ddlat_raw, ddlon_raw, ddalt_raw])))

        if len(gps_buffer) > gps_delay:
            pos_meas = gps_buffer.pop(0)
        else:
            pos_meas = gps_buffer[0]

        if len(imu_buffer) > imu_delay:
            vel_meas, acc_meas = imu_buffer.pop(0)
        else:
            vel_meas, acc_meas = imu_buffer[0]
        # ekf update loop
        ekf.predict(acc_meas)
        ekf.update_pos(pos_meas)
        ekf.update_vel(vel_meas)

        true_log.append([gps.latitude, gps.longitude, gps.altitude])
        est_log.append(ekf.x[:3,0].copy())
        time_log.append(t)
        P_log.append(ekf.P.copy())

        if t >= next_log_time:
            error_log.append(est_log[-1] - true_log[-1])
            next_log_time += log_interval
        # move follower drone to estimated location
        futF = cli.moveToGPSAsync(ekf.x[0,0], ekf.x[1,0], ekf.x[2,0], velocity=1.5, timeout_sec=dt+0.02,
                                  drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                  yaw_mode=airsim.YawMode(False,0), vehicle_name="Drone1")

        futL.join(); futF.join()
        lag = time.time()-tic
        if lag < dt: time.sleep(dt - lag)
        t += dt

    for d in ["Drone2","Drone1"]:
        cli.armDisarm(False,d); cli.enableApiControl(False,d)

    true_arr = np.array(true_log)
    est_arr = np.array(est_log)
    t_arr = np.array(time_log)
    error_arr = np.array(error_log)
    # store results
    save_results(true_arr, est_arr, t_arr, P_log, error_arr, gps_rate, noise_type, exp_name)

def save_results(true_arr, est_arr, t_arr, P_log, error_arr, gps_rate, noise_type, exp_name):
    label = f"{exp_name}_gps{gps_rate}_noise{noise_type}"
    out_dir = os.path.join(SAVE_PATH, label)
    os.makedirs(out_dir, exist_ok=True)

    var_names = ["Latitude (deg)", "Longitude (deg)", "Altitude (m)"]

    plt.figure(figsize=(12, 8))

    for i in range(3):
        ax = plt.subplot(3, 1, i+1)
        ax.plot(t_arr, true_arr[:,i], label="True Trajectory")
        ax.plot(t_arr, est_arr[:,i], '--', label="EKF Estimate")
        ax.set_ylabel(var_names[i])
        ax.grid(True)
        if i == 0:
            ax.legend()
        if i == 2:
            ax.set_xlabel("Time (s)")

    plt.suptitle(f"Trajectory Tracking (GPS {gps_rate} Hz, Noise: {noise_type})\nNumerics: Track L Posn ← GPS → Measure err, Mean, Max")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "trajectory_all_in_one.png"), dpi=300)
    plt.close()

    # Error metrics
    lat_mean_rad = np.mean(true_arr[:,0]) * DEG2RAD
    lat_err_m = (est_arr[:,0] - true_arr[:,0]) * (EARTH_R * DEG2RAD)
    lon_err_m = (est_arr[:,1] - true_arr[:,1]) * (EARTH_R * math.cos(lat_mean_rad) * DEG2RAD)
    alt_err_m = est_arr[:,2] - true_arr[:,2]

    mean_abs_errors = [np.mean(np.abs(lat_err_m)),
                       np.mean(np.abs(lon_err_m)),
                       np.mean(np.abs(alt_err_m))]
    max_errors = [np.max(np.abs(lat_err_m)),
                  np.max(np.abs(lon_err_m)),
                  np.max(np.abs(alt_err_m))]

    print(f"Mean Abs Error (meters): Lat={mean_abs_errors[0]:.3f}, Lon={mean_abs_errors[1]:.3f}, Alt={mean_abs_errors[2]:.3f}")
    print(f"Max Error (meters): Lat={max_errors[0]:.3f}, Lon={max_errors[1]:.3f}, Alt={max_errors[2]:.3f}")

    # Mean error bar plot
    plt.figure(figsize=(8,4))
    plt.bar(["Lat", "Lon", "Alt"], mean_abs_errors, color='skyblue')
    for i, v in enumerate(mean_abs_errors):
        plt.text(i, v + 0.05, f"{v:.2f} m", ha='center')
    plt.ylabel("Mean Absolute Error (meters)")
    plt.title(f"Mean Absolute Error\n(GPS {gps_rate} Hz, Noise: {noise_type})")
    plt.savefig(os.path.join(out_dir, "mean_abs_error.png"), dpi=300)
    plt.close()

    # Covariance plot
    P_xx = [P[0,0] for P in P_log]
    P_yy = [P[1,1] for P in P_log]
    P_zz = [P[2,2] for P in P_log]

    plt.figure(figsize=(10,6))
    plt.plot(t_arr, P_xx, label="P_xx (Lat Cov)")
    plt.plot(t_arr, P_yy, label="P_yy (Lon Cov)")
    plt.plot(t_arr, P_zz, label="P_zz (Alt Cov)")
    plt.xlabel("Time (s)")
    plt.ylabel("Covariance Diagonal Entry")
    plt.legend()
    plt.grid(True)
    plt.title(f"Covariance Diagonal vs Time")
    plt.savefig(os.path.join(out_dir, "covariance_diagonal.png"), dpi=300)
    plt.close()




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='gps_update', help='Experiment name prefix')
    parser.add_argument('--gps-rate', type=int, default=20, help='GPS update rate (Hz)')
    parser.add_argument('--noise-type', type=str, default='gaussian', choices=['gaussian','cauchy','student'], help='Noise model')
    parser.add_argument('--log-interval', type=float, default=5.0, help='Interval (sec) to log downsampled errors')

    args = parser.parse_args()

    run_ekf_tracking(gps_rate=args.gps_rate,
                     noise_type=args.noise_type,
                     log_interval=args.log_interval,
                     exp_name=args.exp)
