import numpy as np

with open('teapot.obj', 'r') as file:
    input_data = file.readlines()

vertexes = []
facets = []
for temp in input_data:
    if temp[0] == 'v':
        vertex = [float(coordinate) for coordinate in temp.split()[1:]]
        vertexes.append(vertex)
    elif temp[0] == 'f':
        face = [int(i) - 1 for i in temp.split()[1:]]
        facets.append(face)

def area_triangle(vertexes):
    A = np.linalg.norm(vertexes[0] - vertexes[1])
    B = np.linalg.norm(vertexes[1] - vertexes[2])
    C = np.linalg.norm(vertexes[2] - vertexes[0])

    half_meter = (A + B + C) / 2
    area = np.sqrt(half_meter * (half_meter - A) * (half_meter - B) * (half_meter - C))

    return half_meter, area

def inscr_radius(vertexes):
    half_meter, area = area_triangle(vertexes)
    radius = area / half_meter

    return radius

total_area = 0
for face in facets:
    triangle_vertex = np.array([vertexes[i] for i in face])
    radius = inscr_radius(triangle_vertex)
    area = np.pi * radius**2
    total_area += area

print('Суммарная площадь всех вписанных в треугольники окружностей:', total_area)

def largest_cosine(vertexes):
    max_cos = -1
    for i in range(3):
        vec = []
        for j in range(3):
            if i < j:
                vec.append(vertexes[j] - vertexes[i])
            elif i > j:
                vec.append(vertexes[i] - vertexes[j])
            else:
                continue
        cosine = np.dot(vec[1], vec[0]) / (np.linalg.norm(vec[1]) * np.linalg.norm(vec[0]))
        max_cos = max(max_cos, cosine)

    return max_cos

max_cosine = -1
for face in facets:
    triangle_vertex = np.array([vertexes[i] for i in face])
    cosine = largest_cosine(triangle_vertex)
    max_cosine = max(max_cosine, cosine)

print('Самый большой косинус угла:', max_cosine)

