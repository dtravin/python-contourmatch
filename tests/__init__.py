TEMPLATE_HOUSE = 'house'
TEMPLATE_ROMB = 'romb'
TEMPLATE_SHIP = 'ship'
points_house_90 = [[0, 3], [3, 0], [6, 0], [6, 3], [6, 6], [3, 6]]
points_house_90_anticlockwise = list(reversed(points_house_90)) # anti clock-wise rotation
points_house_90_scaled2 = [[0, 6], [6, 0], [12, 0], [12, 6], [12, 12], [6, 12]]
vectors_house_90 = [3+3j,  3+0j,  0-3j,  0-3j, -3.+0j, -3+3j]

points_house_rotated_minus90 = [[6, 3], [3, 6], [0, 6], [0, 3], [0, 0], [3, 0]]

points_romb = [[3, 0], [6, 3], [6, 6], [3, 9], [0, 6], [0, 3]]
points_ship = [[3, 0], [6, 0], [6, 3], [9, 3], [12, 3], [9, 6], [6, 6], [3, 6],
               [0, 6], [0, 3], [3, 3]]

points_unequalized = [[0, 0], [3, 4], [6, 8], [9, 12], [12, 16], [15, 20], [18, 24], [21, 28], [24, 32], [27, 36], [30, 40],
    [60, 80], [0, 80],
]

points_equalized_24 = [
    [0, 0], [6, 8], [12, 16], [18, 24], [24, 32], [30, 40],
    [36, 48], [42, 56], [48, 64], [54, 72], [60, 80],
    [50, 80], [40, 80], [30, 80], [20, 80], [10, 80], [0, 80],
    [0, 70], [0, 60], [0, 50], [0, 40], [0, 30], [0, 20], [0, 10]
]

