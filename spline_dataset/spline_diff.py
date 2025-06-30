import numpy as np
import matplotlib.pyplot as plt

def generate_imu_data(data, sampling_rate=100):
    """
    Calculate IMU data from a given 2D spline trajectory.
    IMU data includes:
        - Accelerometer: projection of acceleration on tangent and normal vectors.
        - Gyroscope: angular velocity computed from change in heading.
    """
    dt = 1.0 / sampling_rate

    # Velocity vector
    velocity = np.diff(data, axis=0) / dt
    vel_abs = np.linalg.norm(velocity, axis=1)

    # Unit tangent vectors
    tau = velocity / vel_abs[:, None]

    # Unit normal vectors (perpendicular to tau)
    n = tau[:, ::-1].copy()
    n[:, 0] = -n[:, 0]

    # Acceleration vector
    acc = np.diff(velocity, axis=0) / dt #TODO remove projection, return acc 

    # Accelerometer measurements (projected on tau and n)
    acc_measurements = np.zeros_like(acc)
    for i in range(len(acc)):
        acc_measurements[i, 0] = np.dot(acc[i], tau[i])
        acc_measurements[i, 1] = np.dot(acc[i], n[i])

    acc_measurements = acc

    # Gyroscope measurements (angular velocity around Z-axis)
    gyro_measurements = np.zeros(len(tau) - 1)
    yaw = np.zeros(len(tau)-1)
    for i in range(len(gyro_measurements)):
        cross = np.cross(tau[i], tau[i + 1])
        dot = np.dot(tau[i], tau[i + 1])
        angle = np.arctan2(cross, dot)
        gyro_measurements[i] = angle / dt #совпадает ось Z 

    # Align lengths
    N = min(len(acc_measurements), len(gyro_measurements))
    acc_measurements = acc_measurements[:N]
    gyro_measurements = gyro_measurements[:N]
    tau = tau[:N]
    n = n[:N]
    velocity = velocity[:N]
    time = dt * np.arange(N)
    
    yaw = np.arctan2(tau[:, 1], tau[:, 0])
    
    poses = np.concatenate([data[1:N+1], yaw[:N, None]], axis=1)
    
    assert acc_measurements.shape[0] == gyro_measurements.shape[0], "Mismatch between acc and gyro lengths"

    return acc_measurements, gyro_measurements.reshape(-1, 1), velocity, poses, tau


def visualize_imu_data(data, acc_measurements, gyro_measurements):
    fig, ax1 = plt.subplots()

    ax1.set_title("IMU Measurements")
    ax1.set_xlabel("Time index")
    ax1.plot(acc_measurements[:, 0], label='Acc X')
    ax1.plot(acc_measurements[:, 1], label='Acc Y')
    ax1.set_ylabel("Acceleration")
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(gyro_measurements, color='green', label='Omega Z')
    ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.legend(loc='upper right')

    plt.figure()
    plt.title("Trajectory")
    plt.plot(data[:, 0], data[:, 1])
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()


if __name__ == "__main__":
    log_path = './out/splines_/spline_0.txt'
    data = np.genfromtxt(log_path)

    acc_measurements, gyro_measurements, tau, n, velocity, time = generate_imu_data(data)

    visualize_imu_data(data, acc_measurements, gyro_measurements)