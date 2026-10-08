import random

import numpy

# COMMON VARIABLES

SYSTEM_RANDOM = random.SystemRandom()  # ... Not available on all systems ... (Python official doc)

# simulator activators (environment conditions)

FIXED_WIND = False
ACTIVATE_SMOKE = False
ACTIVATE_WIND = False
# To avoid throwing "KeyError: 'Layer'" when prob burning maps are shown (so UAV won't get its "Layer" attribute in the
# "portrayal_method(obj)"), NUM_AGENTS must be set to 0.
PROBABILITY_MAP = False

# model params specifications

BATCH_SIZE = 90
WIDTH = 50  # in python [height, width] for grid, in js [width, heigh]
HEIGHT = 50
BURNING_RATE = 1
FIRE_SPREAD_SPEED = 2
FUEL_UPPER_LIMIT = 10
FUEL_BOTTOM_LIMIT = 7

DENSITY_PROB = 1  # Tree density (Float number in the interval [0, 1])

WIND_DIRECTION = "south"
# if FIXED_WIND == False (compose wind), then variables inside the if statement are set to be used in the project
if not FIXED_WIND:
    # Possible mixed wind directions: NW, NE, SW, SE"
    FIRST_DIR = "south"  # Introduce first wind direction (north, south, east, west):
    SECOND_DIR = (
        "east"  # Introduce second wind direction (probability calculated based on first one),
    )
    FIRST_DIR_PROB = 0.8  # Introduce first wind probability [0, 1]
MU = 0.9  # Wind velocity (Float number in the interval [0, 1])

SMOKE_PRE_DISPELLING_COUNTER = 2

# UAVs params

NUM_AGENTS = 2  # Set to 2 for testing/external control
N_ACTIONS = 4
UAV_OBSERVATION_RADIUS = 8
side = (UAV_OBSERVATION_RADIUS * 2) + 1
N_OBSERVATIONS = side * side
SECURITY_DISTANCE = 10

# colors

VEGETATION_COLORS = [
    "#414141",
    "#9eff89",
    "#85e370",
    "#72d05c",
    "#62c14c",
    "#459f30",
    "#389023",
    "#2f831b",
    "#236f11",
    "#1c630b",
    "#175808",
    "#124b05",
]
FIRE_COLORS = [
    "#414141",
    "#d8d675",
    "#eae740",
    "#fefa01",
    "#fed401",
    "#feaa01",
    "#fe7001",
    "#fe5501",
    "#fe3e01",
    "#fe2f01",
    "#fe2301",
    "#fe0101",
]
SMOKE_COLORS = ["#ababab"]
BLACK_AND_WHITE_COLORS = [
    "#ffffff",
    "#e6e6e6",
    "#c9c9c9",
    "#b1b1b1",
    "#a1a1a1",
    "#818181",
    "#636363",
    "#474747",
    "#303030",
    "#1a1a1a",
    "#000000",
]
COLORS_LEN = len(VEGETATION_COLORS)


# functions


# function that normalize fuel values to fit them with vegetation and fire colors
def normalize_fuel_values(fuel, limit):
    if fuel > limit:
        fuel = limit
    return max(0, round((fuel / limit) * COLORS_LEN - 1))


# function that normalize any number into a desired range
def normalize(to_normalize, upper, multiplier, subtractor):
    return ((to_normalize / upper) * multiplier) - subtractor


# function that calculates the Euclidean distance between two certain positions
def euclidean_distance(x1, y1, x2, y2):
    a = numpy.array((x1, y1))
    b = numpy.array((x2, y2))
    dist = numpy.linalg.norm(a - b)
    return dist


# function that calculates the grade of influence of cell s' over cell s, based on a distance_limit
def distance_rate(s, s_, distance_limit):
    m_d = euclidean_distance(s[0], s[1], s_[0], s_[1])
    result = 0.0
    if 0.0 < m_d <= distance_limit:
        result = m_d**-2.0
    return result
