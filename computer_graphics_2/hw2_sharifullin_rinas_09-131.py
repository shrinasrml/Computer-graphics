import numpy as np
import matplotlib.pyplot as plt
from screeninfo import get_monitors

def read_file(file_path):
    with open(file_path, 'r') as file:
        input_data = file.readlines()

    vertexes = []
    facets = []
    for temp in input_data:
        if temp[0] == 'v':
            vertex = np.array([float(coordinate) for coordinate in temp.split()[1:]])
            vertexes.append(vertex)
        elif temp[0] == 'f':
            face = [int(i) - 1 for i in temp.split()[1:]]
            facets.append(face)

    return np.array(vertexes), np.array(facets)

def get_screen_size():
    monitors = get_monitors()

    if monitors:
        monitor = monitors[0]
        scr_width = monitor.width // 3
        scr_height = monitor.height // 3
        return np.array([scr_width, scr_height])
    else:
        print('Мониторы не найдены!')
        return None

def scale_and_transfer(vertexes, screen_size):
    max_values = np.max(vertexes[:, :3], axis=0)

    coeff = screen_size[1] / 2 / max_values[1]
    scale_vertexes = vertexes[:, :3] * coeff

    transfer_vector = np.array([screen_size[0] / 2, screen_size[1] / 2, 0])
    new_vertexes = scale_vertexes + transfer_vector

    return new_vertexes

def create_image(height, width, background_color):
    image = np.zeros((height, width, 4), np.uint8)
    image[:, :, :3] = background_color
    image[:, :, 3] = 255

    return image

def draw_line(image, point1, point2, gradient):
    x1, y1 = map(int, point1[:2])
    x2, y2 = map(int, point2[:2])

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if x1 < x2:
        sx = 1
    else:
        sx = -1

    if y1 < y2:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    while True:
        if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]:
            color = gradient[x1]
            image[image.shape[0] - 1 - y1, x1, :3] = color
            image[image.shape[0] - 1 - y1, x1, 3] = 255

        if x1 == x2 and y1 == y2:
            break

        err2 = 2 * err

        if err2 > -dy:
            err -= dy
            x1 += sx

        if err2 < dx:
            err += dx
            y1 += sy

def draw_triangle(image, vertexes, color):
    for i in range(3):
        draw_line(image, vertexes[i], vertexes[(i + 1) % 3], color)

def generate_gradient(image_width):
    gradient = np.zeros((image_width, 3), dtype=np.uint8)

    start_color = np.array([255, 0, 0], dtype=np.uint8)
    end_color = np.array([255, 255, 0], dtype=np.uint8)

    step = (end_color - start_color) / (image_width - 1)

    for i in range(image_width):
        gradient[i] = start_color + i * step

    return gradient

def show_image(img):
    plt.figure()
    plt.imshow(img)
    plt.savefig('teapot_image.png')
    plt.show()

file = 'teapot.obj'
vertexes, facets = read_file(file)

screen_size = get_screen_size()

scaled_vertexes = scale_and_transfer(vertexes, screen_size)

image = create_image(screen_size[1], screen_size[0], 3)

gradient = generate_gradient(screen_size[0])

for face in facets:
    triangle_vertexes = scaled_vertexes[face]
    draw_triangle(image, triangle_vertexes, gradient)

show_image(image)

