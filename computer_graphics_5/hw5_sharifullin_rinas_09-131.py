import numpy as np
import matplotlib.pyplot as plt
import PIL
import math

file = 'african_head.obj'
def read_file():
    v_arr, f_arr, vt_arr, vn_arr = [], [], [], []

    file_open = open(file, 'r')
    for str in file_open:
        if str[0] == 'v' and str[1] == ' ':
            v_arr.append(str.strip('\n')[2:])
        elif str[0] == 'v' and str[1] == 't':
            vt_arr.append(str.strip('\n')[2:])
        elif str[0] == 'v' and str[1] == 'n':
            vn_arr.append(str.strip('\n')[2:])
        elif str[0] == 'f':
            f_arr.append(str.strip('\n')[2:])
    file_open.close()

    v, f, vt, vn = [], [], [], []
    f = f_arr

    for i in range(len(v_arr)):
        v.append(list(map(float, v_arr[i].split())))

    for i in range(len(vt_arr)):
        vt.append(list(map(float, vt_arr[i].split())))

    for i in range(len(vn_arr)):
        vn.append(list(map(float, vn_arr[i].split())))

    return np.array(v), f, np.array(vn), np.array(vt)

def get_F_ARR(f_arr):
    f_arr = list(f_arr)
    f = []

    for i in range(len(f_arr)):
        f.append(f_arr[i].split())

    f_arr = []
    for i in range(len(f)):
        f1 = [
            int(f[i][0].split('/')[0]),
            int(f[i][0].split('/')[1]),
            int(f[i][0].split('/')[2])
        ]
        f2 = [
            int(f[i][1].split('/')[0]),
            int(f[i][1].split('/')[1]),
            int(f[i][1].split('/')[2])
        ]
        f3 = [
            int(f[i][2].split('/')[0]),
            int(f[i][2].split('/')[1]),
            int(f[i][2].split('/')[2])
        ]
        f_arr.append([f1, f2, f3])

    return np.array(f_arr)

def rot_matrix(axis, angle):
    axis = np.asarray(axis)
    angle = (angle / 180) * np.pi
    axis = axis / np.linalg.norm(axis)

    cos_tetta = np.cos(angle)
    sin_tetta = np.sin(angle)
    one_minus_cos_tetta = 1 - cos_tetta

    x, y, z = axis
    x_sin_tetta = x * sin_tetta
    y_sin_tetta = y * sin_tetta
    z_sin_tetta = z * sin_tetta
    x_y_one_minus_cos_theta = x * y * one_minus_cos_tetta
    x_z_one_minus_cos_theta = x * z * one_minus_cos_tetta
    y_z_one_minus_cos_theta = y * z * one_minus_cos_tetta

    rotation_matrix = np.array([
        [cos_tetta + x**2 * one_minus_cos_tetta, x_y_one_minus_cos_theta - z_sin_tetta, x_z_one_minus_cos_theta + y_sin_tetta, 0],
        [x_y_one_minus_cos_theta + z_sin_tetta, cos_tetta + y**2 * one_minus_cos_tetta, y_z_one_minus_cos_theta - x_sin_tetta, 0],
        [x_z_one_minus_cos_theta - y_sin_tetta, y_z_one_minus_cos_theta + x_sin_tetta, cos_tetta + z**2 * one_minus_cos_tetta, 0],
        [0, 0, 0, 1]
    ])

    return rotation_matrix

def rotOX(alpha):
    return rot_matrix([1, 0, 0], alpha)

def rotOY(alpha):
    return rot_matrix([0, 1, 0], alpha)

def rotOZ(alpha):
    return rot_matrix([0, 0, 1], alpha)

def rot(ax, ay, az):
    return rot_matrix([ax, ay, az], 1)

def scal_matrix(a, b, c):
    return np.array([[a, 0, 0, 0], [0, b, 0, 0], [0, 0, c, 0], [0, 0, 0, 1]])

def shift_matrix(vec0, vec1, vec2):
    return np.array([[1, 0, 0, vec0], [0, 1, 0, vec1], [0, 0, 1, vec2], [0, 0, 0, 1]])

def Mo2w(vec0, vec1, vec2, ax, ay, az, a, b, c):
    return np.array(shift_matrix(vec0, vec1, vec2) @ rot(ax, ay, az) @ scal_matrix(a, b, c))

def projective_coord(x):
    return np.vstack([x, np.ones_like(x[0])])

def сartesian_coord(x):
    return x[:-1] / x[-1]

def sign(k):
    return -1 if k < 0 else 1

def brezenhem(img, x0, y0, x1, y1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    err = 0
    change = False

    if x0 == x1 and y0 == y1:
        return

    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0

    if y1 - y0 > x1 - x0:
        x0, y0, x1, y1 = y0, x0, y1, x1
        change = True

    if abs(y1 - y0) > abs(x1 - x0):
        x0, y0, x1, y1 = y1, x1, y0, x0
        change = True

    delta = abs((y1 - y0) / (x1 - x0))
    y = y0

    for x in range(x0, x1 + 1):
        if x>0 and y > 0:
            if not change:
                img[800 - y, 800 - x] = color
            else:
                img[800 - x, 800 - y] = color
        err = err + delta
        if err >= 0.5:
            y = y + sign(y1 - y0)
            err = err - 1

def draw(img, v, f):
    f_ = f[:, :, 0]
    for i in range(len(f_)):
        for j in range(len(f_[i])):
            for k in range(j, len(f_[i]) - 1):
                brezenhem(img, v[f_[i][j] - 1][0], v[f_[i][j] - 1][1], v[f_[i][k + 1] - 1][0], v[f_[i][k + 1] - 1][1])

def normal_vec(vec):
    length = np.linalg.norm(vec)
    return vec / length

def Mw2c(point1, point2):
    direction_vector = normal_vec(np.array([point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]]))
    camera_up = np.array([0, 1, 0])
    camera_right = np.cross(camera_up, direction_vector)
    camera_up = np.cross(direction_vector, camera_right)

    transformation_matrix = np.array([
        [camera_right[0], camera_right[1], camera_right[2], 0],
        [camera_up[0], camera_up[1], camera_up[2], 0],
        [-direction_vector[0], -direction_vector[1], -direction_vector[2], 0],
        [0, 0, 0, 1]
    ])

    return transformation_matrix.T

def Mproj(right, left, top, bottom, near, far):
    A = np.array([
        [2 / (right - left), 0, 0, -((right + left) / (right - left))],
        [0, 2 / (top - bottom), 0, -((top + bottom) / (top - bottom))],
        [0, 0, -(2 / (far - near)), -((far + near) / (far - near))],
        [0, 0, 0, 1]
    ])

    return A

def get_pixels(image_path, count):
    try:
        image = PIL.Image.open(image_path)
        h, w = image.size
        pixels = np.array(image.getdata()).reshape((h, w, count))
        return pixels
    except FileNotFoundError:
        print(f"Файл не найден: {image_path}")

        return None

def Mviewport(x, y, h, w, d):
    mtr = np.array([
                    [w/2, 0, 0, x+w/2],
                    [0, h/2, 0, y+h/2],
                    [0, 0, d/2, d/2],
                    [0, 0, 0, 1]])
    return mtr

def rasterize(p1, p2, p3, P1, P2, P3, pix, z_buffer, image, n):
    xmin = min(p1[0], p2[0], p3[0])
    ymin = min(p1[1], p2[1], p3[1])
    xmax = max(p1[0], p2[0], p3[0])
    ymax = max(p1[1], p2[1], p3[1])

    T = np.linalg.inv([[p1[0], p2[0], p3[0]],
                       [p1[1], p2[1], p3[1]],
                       [1, 1, 1]])

    vt_a = np.array([
        [P1[0], P2[0], P3[0]],
        [P1[1], P2[1], P3[1]],
        [P1[2], P2[2], P3[2]]
    ])

    for i in range(math.floor(xmin), math.ceil(xmax)):
        for j in range(math.floor(ymin), math.ceil(ymax)):
            abc = T @ [i, j, 1]
            if all(c >= 0 for c in abc):
                texture_coords = vt_a @ abc
                z_b = p1[2] * abc[0] + p2[2] * abc[1] + p3[2] * abc[2]
                if z_buffer[i, j] < z_b:
                    z_buffer[i, j] = z_b
                    image[n - j, n - i] = pix[round((1 - texture_coords[1]) * pix.shape[0]),
                                                round((1 - texture_coords[0]) * pix.shape[1])]

def rasterize2(p1, p2, p3, P1, P2, P3, pix, z_buffer, image, n, color):
    xmin = min(p1[0], p2[0], p3[0])
    ymin = min(p1[1], p2[1], p3[1])
    xmax = max(p1[0], p2[0], p3[0])
    ymax = max(p1[1], p2[1], p3[1])

    T = np.linalg.inv([[p1[0], p2[0], p3[0]],
                       [p1[1], p2[1], p3[1]],
                       [1, 1, 1]])

    vt_a = np.array([
        [P1[0], P2[0], P3[0]],
        [P1[1], P2[1], P3[1]],
        [P1[2], P2[2], P3[2]]
    ])

    for i in range(math.floor(xmin), math.ceil(xmax)):
        for j in range(math.floor(ymin), math.ceil(ymax)):
            abc = T @ [i, j, 1]
            if all(c >= 0 for c in abc):
                texture_coords = vt_a @ abc
                z_b = p1[2] * abc[0] + p2[2] * abc[1] + p3[2] * abc[2]
                if z_buffer[i, j] < z_b:
                    z_buffer[i, j] = z_b
                    image[n - j, n - i] = color

def calculate_lighting(N, L, V):
    Iambient = Ia * ka

    cos_theta = np.dot(N, L) / (np.linalg.norm(N) * np.linalg.norm(L))
    Idiffuse = np.clip(Id * kd * cos_theta, 0, 255)

    R = 2 * np.dot(N, L) * N - L
    norm_R = np.linalg.norm(R)
    norm_V = np.linalg.norm(V)

    if norm_R != 0 and norm_V != 0:
        cos_alfa = np.dot(R, V) / (norm_R * norm_V)
        cos_alfa = max(0, cos_alfa)
        Ispecular = np.clip(Is * ks * cos_alfa ** alfa, 0, 255)
    else:
        Ispecular = np.zeros(3)

    I = Iambient + Idiffuse + Ispecular

    return I

def rasterize_lighting(p1, p2, p3, P1, P2, P3, pix, z_buffer, image, n, vn, A):
    xmin = min(p1[0], p2[0], p3[0])
    ymin = min(p1[1], p2[1], p3[1])
    xmax = max(p1[0], p2[0], p3[0])
    ymax = max(p1[1], p2[1], p3[1])

    T = np.linalg.inv([[p1[0], p2[0], p3[0]],
                       [p1[1], p2[1], p3[1]],
                       [1, 1, 1]])

    vt_a = np.array([
        [P1[0], P2[0], P3[0]],
        [P1[1], P2[1], P3[1]],
        [P1[2], P2[2], P3[2]]
    ])

    for i in range(math.floor(xmin), math.ceil(xmax)):
        for j in range(math.floor(ymin), math.ceil(ymax)):
            abc = T @ [i, j, 1]
            if all(c >= 0 for c in abc):
                texture_coords = vt_a @ abc
                z_b = p1[2] * abc[0] + p2[2] * abc[1] + p3[2] * abc[2]
                if z_buffer[i, j] < z_b:
                    z_buffer[i, j] = z_b

                    normal = (
                        abc[0] * vn[f_[i][0] - 1] +
                        abc[1] * vn[f_[i][1] - 1] +
                        abc[2] * vn[f_[i][2] - 1]
                    )
                    normal = normal_vec(normal)

                    L = normal_vec(light_position - np.array([i, j, z_b]))
                    V = normal_vec(A - np.array([i, j, z_b]))
                    I = calculate_lighting(normal, L, V)
                    color = pix[round((1 - texture_coords[1]) * pix.shape[0]),
                                                round((1 - texture_coords[0]) * pix.shape[1])]*I/255
                    color[color>255] = 255

                    image[n - j, n - i] = color


sizes = 800
v_arr, f_arr, vn_arr, vt_arr = read_file()
f_arr = get_F_ARR(f_arr)

f_ = f_arr[:, :, 0]
f2 = f_arr[:, :, 1]

p1 = [2, 2, 2]
p2 = [-2, -2, 0]

mo2w = Mo2w(-2, 3, -2, 5, 15, 10, 0.8, 0.8, 0.8)
v = projective_coord(v_arr.T)
v = mo2w @ v
mw2c = Mw2c(p1, p2)
v = mw2c @ v

l = np.min(v[0])
r = np.max(v[0])
t = np.max(v[1])
b = np.min(v[1])
N = np.min(v[2])
f = np.max(v[2])

v = Mproj(l, r, t, b, N, f) @ v
v = (Mviewport(0, 0, sizes, sizes, 255) @ v).T

img = np.ones((sizes, sizes, 3), dtype=np.uint8) * 255
img_1 = np.ones((sizes, sizes, 3), dtype=np.uint8) * 255
img_2 = np.ones((sizes, sizes, 3), dtype=np.uint8) * 255
img_3 = np.ones((sizes, sizes, 3), dtype=np.uint8) * 255
color = [40, 40, 40]

def get_1model(v_arr, f_arr):
    draw(img_1, v_arr, f_arr)
    plt.figure()
    plt.imshow(img_1)
    plt.show()
    im = PIL.Image.fromarray(img_1)
    im.save('Model1.png')

get_1model(v, f_arr)

z_buf = np.full((sizes, sizes), -10000)

def get_model():
    pix = get_pixels('african_head_diffuse.tga', 3)
    color = np.array([255, 255, 255])
    for i in range(len(f_)):
        v1 = (v[f_[i][2] - 1] - v[f_[i][1] - 1])[:3]
        v2 = (v[f_[i][1] - 1] - v[f_[i][0] - 1])[:3]
        normal = normal_vec(np.cross(v1, v2))
        centr = (v[f_[i][0] - 1] + v[f_[i][1] - 1] + v[f_[i][2] - 1])/3
        vec = normal_vec(p1-centr[:3])
        sc = np.dot(normal, vec)
        if sc > 0:
            rasterize2(v[f_[i][0] - 1], v[f_[i][1] - 1], v[f_[i][2] - 1],
                 vt_arr[f2[i][0] - 1], vt_arr[f2[i][1] - 1],
                 vt_arr[f2[i][2] - 1], pix, z_buf, img_3, sizes, sc*color)

    plt.figure()
    plt.imshow(img_3)
    plt.show()
    im = PIL.Image.fromarray(img_3)
    im.save('Model.png')

get_model()
def get_2model():
    pix = get_pixels('african_head_diffuse.tga', 3)
    for i in range(len(f_)):
        rasterize(v[f_[i][0] - 1], v[f_[i][1] - 1], v[f_[i][2] - 1],
             vt_arr[f2[i][0] - 1], vt_arr[f2[i][1] - 1],
             vt_arr[f2[i][2] - 1], pix, z_buf, img, sizes)

    plt.figure()
    plt.imshow(img)
    plt.show()
    im = PIL.Image.fromarray(img)
    im.save('Model2.png')

get_2model()

Ia = np.array([30, 30, 30])
ka = np.array([0.7, 0.7, 0.7])

Id = np.array([160, 140, 150])
kd = np.array([0.8, 0.1, 0.6])

Is = np.array([200, 120, 170])
ks = np.array([0.2, 0.3, 0.7])
alfa = 1.0

light_position = np.array([-3, 3, 5])

def get_3model(v_arr, f_arr, vn_arr, A):
    pix = get_pixels('african_head_diffuse.tga', 3)
    for i in range(len(f_)):
        rasterize_lighting(v[f_[i][0] - 1], v[f_[i][1] - 1], v[f_[i][2] - 1],
                                vt_arr[f2[i][0] - 1], vt_arr[f2[i][1] - 1],
                                vt_arr[f2[i][2] - 1], pix, z_buf, img_2, sizes, vn_arr, A)

    plt.figure()
    plt.imshow(img_2)
    plt.show()
    im = PIL.Image.fromarray(img_2)
    im.save('Model3.png')

get_3model(v, f_arr, vn_arr, p1)

