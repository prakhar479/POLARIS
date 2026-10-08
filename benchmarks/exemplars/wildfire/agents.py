# python libraries

import functools
import random

import mesa
from common_fixed_variables import *

# own python modules


# Class Fire holds methods for managing Fire agents
class Fire(mesa.Agent):

    # constructor
    def __init__(self, unique_id, model, burning=False):
        super().__init__(unique_id, model)
        self.fuel = random.randint(FUEL_BOTTOM_LIMIT, FUEL_UPPER_LIMIT)
        self.burning = burning
        self.next_burning_state = None
        self.moore = True
        self.radius = 3
        self.selected_dir = 0
        self.steps_counter = 0
        self.cell_prob = 0.0

        # smoke
        self.smoke = Smoke(fire_cell_fuel=self.fuel)

    # checks if the corresponding Fire agent is burning | True if burning, False if not
    def is_burning(self):
        return self.burning

    # get the corresponding Fire agent remaining fuel | Integer value
    def get_fuel(self):
        return round(self.fuel)

    # get the corresponding Fire agent burning probability
    def get_prob(self):
        return self.cell_prob

    # function that calculates probability of cell s being burned in next time step (p_t+1(s))
    def probability_of_fire(self):
        probs = []
        # if at least cell s has some fuel remaining
        if self.fuel > 0:
            # obtains adjacent cells for a given one (self.pos), based on a radius (self.radius)
            adjacent_cells = self.model.grid.get_neighborhood(
                self.pos, moore=self.moore, include_center=False, radius=self.radius
            )

            # iterates through each adjacent cell to calculate cell s probability of being burned
            # based on the adjacent ones
            for adjacent in adjacent_cells:
                # obtains cell content, such as different agents
                agents_in_adjacent = self.model.grid.get_cell_list_contents([adjacent])
                # iterates through each found agent of an adjacent cell
                for agent in agents_in_adjacent:
                    if type(agent) is Fire:
                        adjacent_burning = 1 if agent.is_burning() else 0
                        # calculates partial probability of burning cell s (self.pos), being influenced by adjacent (s')
                        aux_prob = distance_rate(self.pos, adjacent, self.radius) * adjacent_burning
                        # in this if statement, the wind logic occurs, by biasing the burning cell probability
                        if ACTIVATE_WIND and (adjacent_burning == 1):
                            # applies wind to the partial probability
                            aux_prob = self.model.wind.apply_wind(aux_prob, self.pos, agent.pos)
                        probs.append(1 - aux_prob)
            if (
                len(probs) == 0
            ):  # if a low tree density is set, this might happen, so it must be checked
                P = 0
            else:
                P = 1 - functools.reduce(lambda a, b: a * b, probs)
        else:
            P = 0
        return P

    # Mesa framework native method, which is overwritten, necessary for setting next state of the simulation
    def step(self):
        self.steps_counter += 1
        # make fire spread slower
        if self.steps_counter % FIRE_SPREAD_SPEED == 0:
            # if self.steps_counter == 26: # to model how the wind can suddenly change direction
            #     self.model.wind.wind_direction = 'south'
            self.cell_prob = self.probability_of_fire()
            generated = random.random()
            # set next burning state
            if generated < self.cell_prob:
                self.next_burning_state = True
            else:
                self.next_burning_state = False
            # if possible, subtract BURNING_RATE from fuel of the corresponding cell
            if self.burning and self.fuel > 0:
                self.fuel = self.fuel - BURNING_RATE
            # smoke step
            if ACTIVATE_SMOKE:
                self.smoke.smoke_step(self.burning)

    # Mesa framework native method, which is overwritten, necessary for executing changes made in step() method. This
    # logic is required to not update the overall grid state until all cells step() method where executed.
    def advance(self):
        # make fire spread slower
        if self.steps_counter % FIRE_SPREAD_SPEED == 0:
            self.burning = self.next_burning_state


# Class Smoke holds methods for managing smoke functionality
class Smoke:

    # constructor
    def __init__(self, fire_cell_fuel):
        self.smoke = False
        self.dispelling_counter_start_value = fire_cell_fuel
        self.dispelling_lower_bound_start_value = SMOKE_PRE_DISPELLING_COUNTER
        self.dispelling_lower_bound = self.dispelling_lower_bound_start_value
        self.dispelling_counter = self.dispelling_counter_start_value

    # it gets the remaining dispelling counter value
    def get_dispelling_counter_value(self):
        return self.dispelling_counter

    # it gets the remaining pre-dispelling counter value
    def get_dispelling_counter_start_value(self):
        return self.dispelling_counter_start_value

    # it gets if smoke is active | True if active, False if not
    def is_smoke_active(self):
        return self.smoke

    # it subtracts one from dispelling counter value
    def subtract_dispelling_counter(self):
        self.dispelling_counter -= 1

    # function that updates smoke state and its counters based on certain conditions
    def smoke_step(self, burning):
        # if smoke isn't activated yet:
        if not self.smoke and self.dispelling_counter == self.dispelling_counter_start_value:
            # if pre-dispelling smoke counter can start (cell is burning), or if it already started:
            if (
                burning and self.dispelling_lower_bound == self.dispelling_lower_bound_start_value
            ) or (0 < self.dispelling_lower_bound < self.dispelling_lower_bound_start_value):
                # subtract from pre-dispelling counter (on the way to start smoke)
                self.dispelling_lower_bound -= 1
            # if pre-dispelling smoke counter already finished:
            elif self.dispelling_lower_bound == 0:
                # start smoke counter (activate smoke)
                self.smoke = True
        # if smoke can start, or if it already started
        elif self.smoke:
            # if dispelling counter can start, or if it already started
            if 0 < self.dispelling_counter <= self.dispelling_counter_start_value:
                # subtract from dispelling counter
                self.subtract_dispelling_counter()
            # if dispelling counter already finished
            elif self.dispelling_counter == 0:
                # smoke counter is stopped
                self.smoke = False


# Class Wind holds methods for managing wind functionality
class Wind:

    # constructor
    def __init__(self):
        self.wind_direction = WIND_DIRECTION

    # it allows to change wind direction based on FIRST_DIR_PROB value
    def change_direction(self):
        if SYSTEM_RANDOM.random() < FIRST_DIR_PROB:
            self.wind_direction = FIRST_DIR
        else:
            self.wind_direction = SECOND_DIR

    # function to apply wind to partial burning probability of cell s (relative_center_pos),
    # caused by cell s' (adjacent_pos)
    def apply_wind(self, aux_prob, relative_center_pos, adjacent_pos):
        # if wind is compound by more than one direction
        if not FIXED_WIND:
            self.change_direction()
            # print("Wind: ", self.wind_direction)
        if self.is_on_wind_direction(relative_center_pos, adjacent_pos):
            aux_prob = aux_prob + (
                MU * (1 - aux_prob)
            )  # part of 1 I- 'aux_prob' probability is added, depending on mu
        else:
            aux_prob = aux_prob - (
                MU * aux_prob
            )  # part of 'aux_prob' probability is removed, depending on mu
        return aux_prob

    # function that checks if cell located in relative_center_pos is on wind direction, influenced by cell located
    # in adjacent_pos
    def is_on_wind_direction(self, relative_center_pos, adjacent_pos):
        on_wind_direction = False
        if self.wind_direction == "east":
            if (relative_center_pos[0] > adjacent_pos[0]) and (
                relative_center_pos[1] == adjacent_pos[1]
            ):
                on_wind_direction = True
        elif self.wind_direction == "west":
            if (relative_center_pos[0] < adjacent_pos[0]) and (
                relative_center_pos[1] == adjacent_pos[1]
            ):
                on_wind_direction = True
        elif self.wind_direction == "north":
            if (relative_center_pos[1] > adjacent_pos[1]) and (
                relative_center_pos[0] == adjacent_pos[0]
            ):
                on_wind_direction = True
        elif self.wind_direction == "south":
            if (relative_center_pos[1] < adjacent_pos[1]) and (
                relative_center_pos[0] == adjacent_pos[0]
            ):
                on_wind_direction = True
        return on_wind_direction


# Class UAV holds methods for managing UAV agents
class UAV(mesa.Agent):

    # constructor
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.moore = True
        self.selected_dir = 0

    # function that checks if an UAV in a certain position (pos), has another UAV nearby. If so, it can't move,
    # otherwise it will be possible to move.
    def not_UAV_adjacent(self, pos):
        can_move = True
        agents_in_pos = self.model.grid.get_cell_list_contents([pos])
        for agent in agents_in_pos:
            if type(agent) is UAV:
                can_move = False
        return can_move

    # function for obtaining observed cells for the corresponding UAV
    def surrounding_states(self):
        surrounding_states = []
        # obtains adjacent cells s' from a concrete cell s (self.pos)
        adjacent_cells = self.model.grid.get_neighborhood(
            self.pos, moore=self.moore, include_center=True, radius=UAV_OBSERVATION_RADIUS
        )
        # obtains each fire cell state, in a list (1 if its burning, 0 if it isn't)
        for cell in adjacent_cells:
            agents = self.model.grid.get_cell_list_contents([cell])
            for agent in agents:
                if type(agent) is Fire:
                    surrounding_states.append(int(agent.is_burning() is True))
        return surrounding_states

    # function for moving UAV over the grid area
    def move(self):
        # vectors for moving to different positions, based on 4 directions = [0, 1, 2, 3] = [right, down, left, up].
        # For example, if direction 1 is chosen, then the UAV moves 0 cells in x-axis, and -1 cell in y-axis
        move_x = [1, 0, -1, 0]
        move_y = [0, -1, 0, 1]
        moved = False

        # it calculates the position the corresponding UAV will move to
        pos_to_move = (
            self.pos[0] + move_x[self.selected_dir],
            self.pos[1] + move_y[self.selected_dir],
        )
        # checks if the position to move is inside the grid bounds, and that the UAV doesn't have other UAV nearby. If
        # so, the UAV moves
        if not self.model.grid.out_of_bounds(pos_to_move) and self.not_UAV_adjacent(pos_to_move):
            self.model.grid.move_agent(self, tuple(pos_to_move))
            moved = True

        return moved

    # Mesa framework native method, which is overwritten, necessary for executing changes made in step() method
    # (as it can be seen, in this case UAVs don't need to update anything in step() method, so it isn't overwritten).
    def advance(self):
        self.move()
