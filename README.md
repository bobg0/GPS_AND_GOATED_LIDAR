## 1. Environment preparation

### 1.1 Clone repository
```bash
git clone https://github.com/your-repo/airsim-ekf.git
cd airsim-ekf
````

### 1.2 Install AirSim

* Please refer to [AirSim official documentation](https://github.com/microsoft/AirSim) to complete the compilation and installation of AirSim.

* Make sure the Unreal Engine environment is configured and can run the examples.

### 1.3 Replace `settings.json`

Copy the provided `settings.json` file to the root directory of the AirSim project (overwriting the default configuration):

```
<Path_to_AirSim>/Unreal/Environments/Blocks/settings.json
```

This file already contains two drones (Drone1 + Drone2) and their respective sensor configurations.

### 1.4 Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

* airsim
* numpy
* matplotlib
* scipy
* scikit-learn

## 2. Experiment script description

### 2.1 LiDAR experiment: `ekf_modular.py`

9-dimensional constant acceleration EKF tracking based on a single-mounted 512-line LiDAR. Supports different sampling rates and noise models.

| Parameters | Description |
| -------------- | ----------------------------------------- |
| `--exp` | Experiment name prefix, output directory is `ekf_plots/<exp>` |
| `--lidar-rate` | LiDAR update rate (Hz), **only single value is accepted** |
| `--noise` | Noise type: `none`/`gaussian`/`student`/`cauchy` |
| `--runtime` | Total runtime (s), default 60 |
| `--log-int` | Downsampling logging interval (s), default 1 |

> **Tip**: In order to avoid data conflicts caused by AirSim environment reset failure, it is not recommended to pass in a frequency list for scanning at one time; please run the script for each frequency separately and restart the environment.

#### Benchmark experiment

```bash
python ekf_modular.py --exp baseline --lidar-rate 10 --noise none
```

#### Sampling rate sweep

```bash
python ekf_modular.py --exp lidar_5hz --lidar-rate 5 --noise none
python ekf_modular.py --exp lidar_3hz --lidar-rate 3 --noise none
python ekf_modular.py --exp lidar_1hz --lidar-rate 1 --noise none
```

#### Noise robustness test

```bash
python ekf_modular.py --exp noise_gauss --lidar-rate 10 --noise gaussian
python ekf_modular.py --exp noise_student --lidar-rate 10 --noise student
python ekf_modular.py --exp noise_cauchy --lidar-rate 10 --noise cauchy
```

### 2.2 GPS Experiment: `ekf_gps_modular.py`

6D EKF tracking based on GNSS + IMU. Supports different GPS update rates and noise models.

| Parameters | Description |
| ---------------- | -------------------------------------------------------- |
| `--exp` | Experiment name prefix, output directory is `GPS_EKF_PLOTS/<exp>_gps<rate>_noise<type>` |
| `--gps-rate` | GPS update rate (Hz) |
| `--noise-type` | Noise model: `gaussian`/`cauchy`/`student` |
| `--log-interval` | Error downsampling logging interval (s), default 5 |

#### Baseline GPS tracking

```bash
python ekf_gps_modular.py --exp gps_baseline --gps-rate 20 --noise-type gaussian
```

#### Noise and sampling rate test

```bash
python ekf_gps_modular.py --exp gps_gauss_10hz --gps-rate 10 --noise-type gaussian
python ekf_gps_modular.py --exp gps_student_20hz --gps-rate 20 --noise-type student
```

## 3. Output results

* LiDAR experimental results are saved in the `ekf_plots/<exp>/` directory, each subdirectory contains:

* `baseline_traj_X.png`, `baseline_traj_Y.png`, `baseline_traj_Z.png`
* `baseline_err.png`
* `baseline_cov.png`
* GPS experimental results are saved in the `GPS_EKF_PLOTS/<exp>_gps<rate>_noise<type>/` directory, including:

* `trajectory_all_in_one.png`
* `mean_abs_error.png`
* `covariance_diagonal.png`
