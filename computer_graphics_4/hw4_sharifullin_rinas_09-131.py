import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def alg_bezier(t, p0, p1, p2):
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def draw_inf(ax, control_points_list):
    points = []
    for control_points in control_points_list:
        if len(control_points) == 3:
            p0, p1, p2 = np.array(control_points[0]), np.array(control_points[1]), np.array(control_points[2])
            t_values = np.linspace(0, 1, 100)
            curve_points = np.array([alg_bezier(t, p0, p1, p2) for t in t_values])
            points.extend(curve_points)
            ax.plot(curve_points[:, 0], curve_points[:, 1], color='black')

def anim(frame):
    ax.cla()
    ax.set_aspect('equal')
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)

    factor = frame / 60
    if frame <= 60:
        mod_control_points = [
            [(x, y) if (x, y) == (1000.00, 1000.00) else (
            x + (x - 1000.00) * 1.5 * factor, y - (y - 1000.00) * (2 / 3) * factor)
             for x, y in control_point]
            for control_point in control_points_list
        ]
    elif 60 < frame <= 90:
       mod_control_points = [
            [(x, y) if (x, y) == (1000.00, 1000.00) else (
                x + (x - 1000.00) * 1.5, y - (y - 1000.00) * (2 / 3) * factor)
             for x, y in control_point]
            for control_point in control_points_list
        ]
    elif 90 < frame <= 150:
        factor = (150 - frame) / 60
        mod_control_points = [
            [(x, y) if (x, y) == (1000.00, 1000.00) else (
                x + (x - 1000.00) * 1.5 - (x - 1000.00) * 1.5 * (1 - factor) * (2 / 3), y - (y - 1000.00) * factor)
             for x, y in control_point]
            for control_point in control_points_list
        ]
    elif 150 < frame <= 180:
        factor = (frame - 90) / 60
        mod_control_points = [
            [(x, y) if (x, y) == (1000.00, 1000.00) else (
                x + (x - 1000.00) * 1.5 - (x - 1000.00) * 1.5 * factor * (2 / 3), y)
             for x, y in control_point]
            for control_point in control_points_list
        ]
    elif 180 < frame <= 240:
        factor = (frame - 180) / 60
        mod_control_points = [
            [(x, y) if (x, y) == (1000.00, 1000.00) else (
                x - (x - 1000.00) * (2 / 3) * factor, y + (y - 1000.00) * 1.5 * factor)
             for x, y in control_point]
            for control_point in control_points_list
        ]
    elif 240 < frame <= 300:
        factor = (frame - 180) / 60
        mod_control_points = [
            [(x, y) if (x, y) == (1000.00, 1000.00) else (
                x - (x - 1000.00) * (2 / 3) * factor, y + (y - 1000.00) * 1.5)
             for x, y in control_point]
            for control_point in control_points_list
        ]
    elif 300 < frame <= 360:
        factor = (360 - frame) / 60
        mod_control_points = [
            [(x, y) if (x, y) == (1000.00, 1000.00) else (
                x - (x - 1000.00) * (2 / 3) + (x - 1000.00) * factor, y + (y - 1000.00) * 1.5 - (y - 1000.00) * 1.5 * (1 - factor) * (2 / 3))
             for x, y in control_point]
            for control_point in control_points_list
        ]
    elif 360 < frame <= 420:
        factor = (frame - 360) / 60
        mod_control_points = [
            [(x, y) if (x, y) == (1000.00, 1000.00) else (
                x, y + (y - 1000.00) * 1.5 - (y - 1000.00) * 1.5 * factor * (2 / 3))
             for x, y in control_point]
            for control_point in control_points_list
        ]

    draw_inf(ax, mod_control_points)

    ax.invert_yaxis()

fig, ax = plt.subplots()

control_points_list = [
    [(1000.00, 1000.00), (1130.68, 888.17), (1204.19, 890.34)],
    [(1204.19, 890.34), (1294.01, 893.34), (1298.21, 1000.00)],
    [(1298.21, 1000.00), (1294.01, 1106.66), (1204.19, 1109.66)],
    [(1204.19, 1109.66), (1130.68, 1111.83), (1000.00, 1000.00)],
    [(1000.00, 1000.00), (869.32, 888.17), (795.81, 890.34)],
    [(795.81, 890.34), (705.99, 893.34), (701.79, 1000.00)],
    [(701.79, 1000.00), (705.99, 1106.66), (795.81, 1109.66)],
    [(795.81, 1109.66), (869.32, 1111.83), (1000.00, 1000.00)]
]

animation = FuncAnimation(fig, anim, frames=np.arange(0, 420, 1), interval=50, repeat=True)
animation.save('inf_animation.gif', writer='pillow', fps=30)

plt.show()
