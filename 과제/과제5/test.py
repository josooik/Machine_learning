import random

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
ACTIONS = (LEFT, RIGHT, UP, DOWN)

EMPTY = 0
TRAP = 1
GOAL = 2

class Robot:
    def __init__(self, initial_x, initial_y):
        self._initial_x = initial_x
        self._initial_y = initial_y
        self.reset()

    def reset(self):
        self._old_x = self._x = self._initial_x
        self._old_y = self._y = self._initial_y

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_old_x(self):
        return self._old_x

    def get_old_y(self):
        return self._old_y

    def move(self, action):
        self._old_x = self._x
        self._old_y = self._y
        if action == LEFT: self._x -= 1
        elif action == RIGHT: self._x += 1
        elif action == UP: self._y -= 1
        else: self._y += 1

class Grid:
    def __init__(self, cols, rows):
        self._cols = cols
        self._rows = rows
        self._grid = [['.'] * cols for _ in range(rows) ]
        self._trap = []
        self._goal = []

    def reset(self):
        for y in range(self._rows):
            for x in range(self._cols):
                self._grid[y][x] = '.'
        for x, y in self._trap:
            self._grid[y][x] = 'X'
        for x, y in self._goal:
            self._grid[y][x] = 'O'

    def add_trap(self, x, y):
        self._trap.append((x, y))

    def add_goal(self, x, y):
        self._goal.append((x, y))

    def update_robot(self, robot):
        x = robot.get_old_x()
        y = robot.get_old_y()
        self._grid[y][x] = '.'
        x = robot.get_x()
        y = robot.get_y()
        self._grid[y][x] = '#'

    def check(self, robot):
        rx = robot.get_x()
        ry = robot.get_y()
        for x, y in self._goal:
            if x == rx and y == ry: return GOAL
        for x, y in self._trap:
            if x == rx and y == ry: return TRAP
        return EMPTY

    def print(self):
        for line in self._grid:
            print(*line)
        print('-' * 10)


class Q:
    def __init__(self, cols, rows):
        self._cols = cols
        self._rows = rows
        self._q = [ [None] * cols for _ in range(rows) ]
        for y in range(rows):
            for x in range(cols):
                self._q[y][x] = [0] * len(ACTIONS)

    def is_valid_action(self, x, y, action):
        if action == LEFT: return x > 0
        elif action == RIGHT: return x < self._cols - 1
        elif action == UP: return y > 0
        else: return y < self._rows -1

    def get_max_q_action(self, x, y):
        valid_actions = []
        valid_q_values = []
        for a in ACTIONS:
            if self.is_valid_action(x, y, a):
                valid_actions.append(a)
                valid_q_values.append(self._q[y][x][a])

        q_max = max(valid_q_values)
        candidates = []
        for a in valid_actions:
            if self._q[y][x][a] == q_max:
                candidates.append(a)
        return random.choice(candidates)

    def get_max_q(self, x, y):
        q_values = []
        for a in ACTIONS:
            if self.is_valid_action(x, y, a):
                q_values.append(self._q[y][x][a])
        return max(q_values)

    def get_next_max_q(self, x, y, action):
        if self.is_valid_action(x, y, action):
            if action == LEFT: return self.get_max_q(x-1, y)
            elif action == RIGHT: return self.get_max_q(x+1, y)
            elif action == UP: return self.get_max_q(x, y-1)
            else: return self.get_max_q(x, y+1)
        return 0

    def set_q_value(self, x, y, action, value):
        self._q[y][x][action] = value

    def print(self):
        print('Q:')
        for line in self._q:
            print('-' * 120)
            print(' |'.join([' '.join(['{:>6.3f}'] * len(x)).format(*x) for x in line]))
        print('-' * 120)

robot = Robot(0, 3)
grid = Grid(4, 4)
grid.add_trap(1, 1)
grid.add_trap(2, 2)
grid.add_goal(3, 0)
grid.update_robot(robot)
q = Q(4, 4)
total_counts = []
move_count = 0

'''
for idx in range(20):
    print('Episode {}'.format(idx))
    robot.reset()
    grid.reset()
    grid.update_robot(robot)
    grid.print()

    while True:
        x = robot.get_x()
        y = robot.get_y()
        action = q.get_max_q_action(x, y)
        next_max_q = q.get_next_max_q(x, y, action)

        robot.move(action)
        move_count += 1

        type = grid.check(robot)
        reward = 0
        if type == GOAL: reward = 1
        elif type == TRAP: reward = -1

        q.set_q_value(x, y, action, reward + 0.9 * next_max_q)

        grid.update_robot(robot)
        grid.print()

        if type == GOAL:
            total_counts.append(move_count)
            move_count = 0
            print('Counts: {}'.format(total_counts))
            break
        elif type == TRAP:
            break
q.print()

'''
for idx in range(20):
    print('Episode {}'.format(idx))
    robot.reset()
    grid.reset()
    grid.update_robot(robot)
    grid.print()

    while True:
        x = robot.get_x()
        y = robot.get_y()
        if random.random() < 0.1:
            candidates = []
            for a in ACTIONS:
                if q.is_valid_action(x, y, a):
                    candidates.append(a)
            action = random.choice(candidates)
        else:
            action = q.get_max_q_action(x, y)
        next_max_q = q.get_next_max_q(x, y, action)

        robot.move(action)
        move_count += 1

        type = grid.check(robot)
        reward = 0
        if type == GOAL: reward = 1
        elif type == TRAP: reward = -1

        q.set_q_value(x, y, action, reward + 0.9 * next_max_q)

        grid.update_robot(robot)
        grid.print()

        if type == GOAL:
            total_counts.append(move_count)
            move_count = 0
            print('Counts: {}'.format(total_counts))
            break
        elif type == TRAP:
            break
q.print()