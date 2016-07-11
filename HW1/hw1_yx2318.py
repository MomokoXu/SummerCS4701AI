import math
import random
import time
import Queue
import numpy as np
import resource



NO_OF_NODE_EXPANDED_BFS = 0
RUNNING_TIME_BFS = 0
MAX_DEPTH_QUEUE = 0
NO_OF_NODE_EXPANDED_DFS = 0
RUNNING_TIME_DFS = 0
MAX_DEPTH_STACK = 0
NO_OF_NODE_EXPANDED_ASTAR = 0
RUNNING_TIME_ASTAR = 0
MAX_DEPTH_ASTAR = 0
MEMORY_BFS = 0 
MEMORY_DFS = 0
MEMORY_ASTAR = 0
MB = 1048576.0


############ State Class ##############
class State(object):
    def __init__(self, n, state, depth):
        self.n_size = n
        self.path = []
        self.total_size = pow(self.n_size, 2)
        self.goal = range(0, self.total_size)
        self.state = state
        self.depth = depth



    # from the parent state, get child states
    def get_child_states(self):
        
        zero_pos = self.state.index(0)
        poss_zero_moves = self.get_possible_moves(zero_pos)
        possible_states = []
        for move in poss_zero_moves:
            next_state = self.state[:]
            (next_state[zero_pos + move], next_state[zero_pos]) = (next_state[zero_pos], next_state[zero_pos + 
                    move])

            next_state_obj = State(n, next_state, self.depth + 1)
            next_state_obj.path = list(self.path)
            if (move == -1):
                direction = 'LEFT'
            elif (move == 1):
                direction = 'RIGHT'
            elif (move == -self.n_size):
                direction = 'UP'
            elif (move == self.n_size):
                direction = 'DOWN'
            next_state_obj.path.append(direction)
            possible_states.append(next_state_obj)
        return possible_states

    # from current zero's position, get possible directions for zero to swap 
    def get_possible_moves(self, position):

        directions = [self.n_size, 1, -1, -self.n_size]#

        valid_moves = []
        for m in directions:
            if (position + m >= 0) and (position + m < self.total_size):
                if m == 1 and position in range(self.n_size - 1, self.total_size, self.n_size):
                    continue
                if m == -1 and position in range(0, self.total_size, self.n_size):
                    continue
                valid_moves.append(m)
        return valid_moves  


# generate the key for the state with the same number except key is the integer
    def generate_state_key(self):
        key = self.state[0] * pow(10, self.total_size - 1)
        for i in range(0, self.total_size - 1):
            key += self.state[self.total_size - 1 - i] * pow(10, i)
        return key

#check is there a solution or not for some cases
    def check_solvable(self):
        cnt = 0
        for i in range(self.total_size):
            if self.state[i] == 0:
                continue
            for j in range(i + 1, self.total_size):
                if self.state[j] == 0:
                    continue
                if self.state[i] > self.state[j]:
                    cnt += 1
        return cnt % 2 == 0

# print the state in matrix form as the n * n board
    def print_state(self):
        for (index, value) in enumerate(self.state):
            print ' %s ' % value, 
            if index in [x for x in range(self.n_size - 1, self.total_size, self.n_size)]:
                print 
        print 



# randomly generate a start state
def generate_start_state_list(n, total_size):
    return list(np.random.permutation(range(total_size)))


#############  BFS  ################
def bfs(n, st):
    if not st.check_solvable():
        print 'Not solvable'
        return

    start_time = time.time()
    queue = Queue.Queue()
    queue.put(st)
    visited = set()
    global RUNNING_TIME_BFS 
    global NO_OF_NODE_EXPANDED_BFS 
    global MAX_DEPTH_QUEUE
    global MEMORY_BFS


    while not queue.empty():
        next_st = queue.get()
        if next_st.depth > MAX_DEPTH_QUEUE:
            MAX_DEPTH_QUEUE = next_st.depth
        if next_st.state == next_st.goal:
            print 'Found'
            RUNNING_TIME_BFS = time.time() - start_time
            print 'The solution:'
            print next_st.path
            print 'Runing time:', '\t', RUNNING_TIME_BFS,'s'
            print 'No of expanded nodes:', '\t', NO_OF_NODE_EXPANDED_BFS
            print 'The cost of path:', '\t', len(next_st.path)
            print 'The maximum depth of the queue:', '\t', MAX_DEPTH_QUEUE
            MEMORY_BFS = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB
            print 'Memory requirments:', '\t', MEMORY_BFS, 'MB'

            return next_st.path
        if next_st.generate_state_key() in visited:
            continue
        NO_OF_NODE_EXPANDED_BFS += 1
        visited.add(next_st.generate_state_key())            
        states = next_st.get_child_states()
        for state in states:
            queue.put(state)
    print 'Not found'
    RUNNING_TIME_BFS = time.time() - start_time
    print 'Runing time:', '\t', RUNNING_TIME_BFS
    print 'No of expanded nodes:', '\t', NO_OF_NODE_EXPANDED_BFS
    print 'The maximum depth of the queue:', '\t', MAX_DEPTH_QUEUE
    MEMORY_BFS = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB
    print 'Memory requirments:', '\t', MEMORY_BFS, 'MB'


############  DFS  ##################

def dfs(n, st):
    if not st.check_solvable():
        print 'Not solvable'
        return

    #stack = Queue.LifoQueue()
    stack = list()
    #stack.put(st)
    stack.append(st)
    visited = set()
    start_time = time.time()

    global RUNNING_TIME_DFS
    global NO_OF_NODE_EXPANDED_DFS 
    global MAX_DEPTH_STACK
    global MEMORY_DFS
    while stack:

        next_st = stack.pop()
        if next_st.depth > MAX_DEPTH_STACK:
            MAX_DEPTH_STACK = next_st.depth

        if next_st.state == next_st.goal:
            print 'Found'
            RUNNING_TIME_DFS = time.time() - start_time
            print 'The solution:'
            print next_st.path
            print 'Runing time:', '\t', RUNNING_TIME_DFS, 's'
            print 'No of expanded nodes:', '\t', NO_OF_NODE_EXPANDED_DFS
            print 'The cost of path:', '\t', len(next_st.path)
            print 'The maximum depth of the stack:', '\t', MAX_DEPTH_STACK
            MEMORY_DFS = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB - MEMORY_BFS
            print 'Memory requirments:', '\t', MEMORY_DFS, 'MB'

            return next_st.path

        if next_st.generate_state_key() in visited:
            continue
        NO_OF_NODE_EXPANDED_DFS += 1
        visited.add(next_st.generate_state_key())
        states = next_st.get_child_states()
        for state in states:
            stack.append(state)
    print 'Not found'
    RUNNING_TIME_DFS = time.time() - start_time
    print 'Runing time:', '\t', RUNNING_TIME_DFS
    print 'No of expanded nodes:', '\t', NO_OF_NODE_EXPANDED_DFS
    print 'The maximum depth of the stack:', '\t', MAX_DEPTH_STACK
    MEMORY_DFS = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB - MEMORY_BFS
    print 'Memory requirments:', '\t', MEMORY_DFS, 'MB'




##############  A star search  ##################

#Priority queue to store states
class PriorityNode(object):
    def __init__(self, priority, state):
        self.priority = priority
        self.state = state
        return
    def __cmp__(self, other):
        return cmp(self.priority, other.priority)

#calculate the manhattan distance as the evaluation function 
def manhattan_distance(n, st):
    man_dist = 0
    for node in st.state:
        if node != 0:
            goal_dist = abs(st.goal.index(node) - st.state.index(node))
            (jumps, steps) = (goal_dist // n, goal_dist % n)
            man_dist += jumps + steps
    return man_dist

# a start search
def a_star(n, st):    
    if not st.check_solvable():
        print 'Not solvable'
        return

    queue = Queue.PriorityQueue()
    queue.put(PriorityNode(0, st))
    visited = set()
    global RUNNING_TIME_ASTAR
    global NO_OF_NODE_EXPANDED_ASTAR
    start_time = time.time()
    global MAX_DEPTH_ASTAR
    global MEMORY_ASTAR
    while not queue.empty():
        next_st = queue.get().state
        if next_st.depth > MAX_DEPTH_ASTAR:
            MAX_DEPTH_ASTAR = next_st.depth
        if next_st.state == next_st.goal:
            print 'Found'
            RUNNING_TIME_ASTAR = time.time() - start_time
            print 'The solution:'
            print next_st.path
            print 'Runing time:', '\t', RUNNING_TIME_ASTAR, 's'
            print 'No of expanded nodes:', '\t', NO_OF_NODE_EXPANDED_ASTAR
            print 'The cost of path', '\t', len(next_st.path)
            print 'The maximum depth of the queue:', '\t', MAX_DEPTH_ASTAR
            MEMORY_ASTAR = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB - MEMORY_DFS
            print 'Memory requirments:', '\t', MEMORY_ASTAR, 'MB'

            return next_st.path
        if next_st.generate_state_key() in visited:
            continue
        NO_OF_NODE_EXPANDED_ASTAR += 1
        visited.add(next_st.generate_state_key())
        states = next_st.get_child_states()
        for state in states:
            cost = len(state.path) + manhattan_distance(n, state)
            queue.put(PriorityNode(cost, state))
    print 'Not found'
    RUNNING_TIME_ASTAR = time.time() - start_time
    print 'Runing time:', '\t', RUNNING_TIME_ASTAR, 's'
    print 'No of expanded nodes:', '\t', NO_OF_NODE_EXPANDED_ASTAR
    print 'The maximum depth of the queue:', '\t', MAX_DEPTH_ASTAR
    MEMORY_ASTAR = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB - MEMORY_DFS
    print 'Memory requirments:', '\t', MEMORY_ASTAR, 'MB'





if __name__ == '__main__':

    n = 3
    start_state = generate_start_state_list(n, n * n)

    #start_state = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    st = State(n, start_state, 0)
    goal = State(n, st.goal, 0)
    print 'start'
    st.print_state()
    print 'goal'
    goal.print_state()
    print

    print 30 * '*', '\t', 'bfs','\t', 30 * '*'
    b = bfs(n, st)
    print 70 * '*'
    print 
    print 30 * '*', '\t', 'dfs','\t', 30 * '*'
    d =dfs(n, st)
    print 70 * '*'

    print 
    print 30 * '*', '\t', 'a-star','\t', 30 * '*'
    a = a_star(n, st)
    print 70 * '*'





