import scipy.interpolate as si
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from matplotlib.backend_bases import MouseButton

def bspline(cv, n=100, degree=3, periodic=False):
    """Calculate n samples on a bspline."""
    cv = np.asarray(cv)
    count = len(cv)

    if count < degree + 1:
        return np.empty((0, 2))  # Not enough points

    if periodic:
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree, 1, degree)
    else:
        degree = np.clip(degree, 1, count - 1)

    kv = (np.arange(0 - degree, count + degree + degree - 1, dtype='int')
          if periodic else
          np.concatenate(([0] * degree, np.arange(count - degree + 1), [count - degree] * degree)))

    u = np.linspace(periodic, count - degree, n)
    return np.array(si.splev(u, (kv, cv.T, degree))).T


def on_move(event):
    if event.inaxes:
        print(f'data coords {event.xdata:.2f} {event.ydata:.2f}, pixel coords {event.x} {event.y}')


def on_close(output_dir):
    if spline_points.shape[0] > 0:
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(f"{output_dir}/spline_points.txt", spline_points)
        print(f"Spline points saved at {output_dir}/spline_points.txt.")


def on_click(event):
    global control_points, spline_points

    if None in [event.xdata, event.ydata]:
        return

    if event.button == MouseButton.LEFT:
        control_points = np.vstack([control_points, [event.xdata, event.ydata]])
    elif event.button == MouseButton.RIGHT and control_points.shape[0] > 0:
        control_points = control_points[:-1]  # Remove last point

    print("Current control points:\n", control_points)

    control[0].set_data(control_points[:, 0], control_points[:, 1])

    spline_points = bspline(control_points, 100 * len(control_points), 4)
    spline[0].set_data(spline_points[:, 0], spline_points[:, 1])

    fig.canvas.draw()


def generate_batch_of_splines(out_path, number_of_splines=10, n_control_points=100, n_pts_spline_segment=100):
    """Generate number_of_splines random splines and save them to files."""
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    for b in range(number_of_splines):
        rnd_pts = np.random.uniform(-5, 5, size=(n_control_points, 2))
        rnd_pts[:, 0] = np.linspace(0, 20, n_control_points)

        spline_points = bspline(rnd_pts, n_control_points * n_pts_spline_segment, 3)
        plt.plot(spline_points[:, 0], spline_points[:, 1], marker='o', markersize=0.5)
        np.savetxt(f'{out_path}/spline_{b}.txt', spline_points)

    plt.grid()
    plt.title('Generated batch of trajectories')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    output_dir = 'out/spline_dataset'
    fig, ax = plt.subplots()
    control_points = np.empty((0, 2))
    spline_points = np.empty((0, 2))

    control = ax.plot([], [], 'x', color='black', label='control points')
    spline = ax.plot([], [], label='spline', color='red')

    ax.axis('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.grid()
    plt.legend()

    plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    plt.connect('close_event', lambda event: on_close(output_dir))
    plt.show()