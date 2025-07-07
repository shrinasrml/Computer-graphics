import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation

def alg_bezier(t, p0, p1, p2):
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def fill_cat(ax, control_points_list):
    cat_points = []
    for control_points in control_points_list:
        if len(control_points) == 3:
            p0, p1, p2 = np.array(control_points[0]), np.array(control_points[1]), np.array(control_points[2])
            t_values = np.linspace(0, 1, 100)
            curve_points = np.array([alg_bezier(t, p0, p1, p2) for t in t_values])
            cat_points.extend(curve_points)

    cat_polygon = Polygon(cat_points, closed=True, edgecolor='none', facecolor='black')
    ax.add_patch(cat_polygon)

dict_frames_eye = {}
def find_frames_eye():
    for i in range(0, 360, 3):
        x_1 = 1026 + 100 * np.cos(np.radians(i))
        y_1 = 584 + 100 * np.sin(np.radians(i))
        x_2 = 1297 + 90 * np.cos(np.radians(i))
        y_2 = 623 + 90 * np.sin(np.radians(i))
        dict_frames_eye[i] = [x_1, y_1, x_2, y_2]

find_frames_eye()
def anim(frame):
    ax.cla()
    ax.set_aspect('equal')
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    fill_cat(ax, control_points_list)

    t = list(np.linspace(2, 0, 60))
    t.extend(list(np.linspace(0, 2, 60)))

    tail_control_points_list = [
        [(760.76, 1580.44), (214.53, 1565 + t[frame//3] * 80.15), (175.91, 1250 + t[frame//3] * 201.74)],
        [(175.91, 1250 + t[frame//3] * 201.74), (235.67, 1135 + t[frame//3] * 97.67), (740.02, 1315.32)]
    ]

    fill_cat(ax, tail_control_points_list)

    ax.add_patch(Circle((1026, 584), 130, color='yellow'))
    ax.add_patch(Circle((1297, 623), 123, color='yellow'))
    ax.add_patch(Circle((dict_frames_eye[frame][0], dict_frames_eye[frame][1]), 30, color='black'))
    ax.add_patch(Circle((dict_frames_eye[frame][2], dict_frames_eye[frame][3]), 30, color='black'))

    ax.invert_yaxis()

fig, ax = plt.subplots()

control_points_list = [
    [(1196.91, 162.16), (1233.98, 229.24), (1263.84, 329.47)],
    [(1263.84, 329.47), (1419.99, 261.99), (1606.18, 249.68)],
    [(1606.18, 249.68), (1646.26, 476.06), (1634.09, 803.09)],
    [(1634.09, 803.09), (2065.83, 1002.39), (1734.88, 1557.27)],
    [(1734.88, 1557.27), (1786.68, 1604.27), (1703.58, 1604.47)],
    [(1703.58, 1604.47), (1681.66, 1681.41), (1626.77, 1634.49)],
    [(1626.77, 1634.49), (1551.33, 1709.18), (1533.22, 1616.47)],
    [(1533.22, 1616.47), (1464.95, 1609.48), (1431.15, 1634.49)],
    [(1431.15, 1634.49), (1409.31, 1713.64), (1346.20, 1691.12)],
    [(1346.20, 1691.12), (1305.56, 1756.45), (1266.41, 1696.27)],
    [(1266.41, 1696.27), (1223.29, 1746.31), (1163.45, 1615.38)],
    [(1163.45, 1615.38), (1032.31, 1734.43), (1024.45, 1670.53)],
    [(1024.45, 1670.53), (913.52, 1716.96), (933.77, 1633.66)],
    [(933.77, 1633.66), (856.31, 1646.04), (895.86, 1564.99)],
    [(895.86, 1564.99), (809.16, 1567.14), (756.76, 1580.44)],
    [(731.02, 1315.32), (531.45, 520.92), (1196.91, 162.16)]
]

animation = FuncAnimation(fig, anim, frames=np.arange(0, 360, 3), interval=50, repeat=True)
animation.save('cat_animation.gif', writer='pillow', fps=30)

plt.show()
