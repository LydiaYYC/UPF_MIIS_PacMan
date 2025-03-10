# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point, PriorityQueue, manhattan_distance
from math import sqrt, log



#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='Offensive', second='Defensive', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########



class RunningPacmanCaptureAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.food_eaten = 0  # to count how many dots we have eaten

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.previous_food_count = len(self.get_food(game_state).as_list())  # record the total food count in the game
        CaptureAgent.register_initial_state(self, game_state)   

    class Node:
        """A* algorithm node class"""
        def __init__(self, position, parent=None, cost=0):
            self.position = position  # Current node position
            self.parent = parent      # Parent node for path tracing
            self.cost = cost          # Actual cost (G value)
            self.heuristic = 0        # Heuristic value (H value)
            self.total = cost + self.heuristic  # Total cost (F value=G+H)
        def __lt__(self, other):
            # Define node comparison for priority queue sorting
            return self.total < other.total
        
    def a_star_search(self, game_state, start, goal):
        """Core logic of A* path search"""
        open_list = PriorityQueue()  # Open list for nodes to explore
        closed_set = set()          # Closed set for visited nodes

        # Initialize start node
        start_node = self.Node(start)
        start_node.heuristic = manhattan_distance(start, goal)  # Calculate H value
        start_node.total = start_node.cost + start_node.heuristic  # Calculate F value
        open_list.push(start_node, start_node.total)  # Add start node to open list

        while not open_list.is_empty():
            # Get node with lowest F value
            current_node = open_list.pop()

            # If goal node is reached, trace back path
            if current_node.position == goal:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]  # Reverse path to get start-to-goal direction

            closed_set.add(current_node.position)  # Mark as visited

            # Generate valid neighboring nodes
            for neighbor in self.get_valid_neighbors(game_state, current_node.position):
                if neighbor in closed_set:
                    continue  # Skip visited nodes

                # Calculate new cost and create neighbor node
                new_cost = current_node.cost + 1
                new_node = self.Node(neighbor, current_node, new_cost)
                new_node.heuristic = manhattan_distance(neighbor, goal)
                new_node.total = new_cost + new_node.heuristic

                # Update open list
                open_list.update(new_node, new_node.total)

        return []  # Return empty list if no path found

    def is_valid_position(self, pos):
        """Check if position is valid"""
        return pos and len(pos) == 2 and isinstance(pos[0], (int, float))

    def get_valid_neighbors(self, game_state, pos):
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [n for n in neighbors if self.is_valid_move(game_state, n)]

    def is_valid_move(self, game_state, pos):
        """Check if move is valid"""
        walls = game_state.get_walls()
        x, y = int(pos[0]), int(pos[1])
        # Check boundary and wall collisions
        if x < 0 or y < 0 or x >= walls.width or y >= walls.height:
            return False
        return not walls[x][y]

    def direction_between(self, from_pos, to_pos):
        """Calculate direction between two positions"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if dx == 1: return Directions.EAST
        if dx == -1: return Directions.WEST
        if dy == 1: return Directions.NORTH
        if dy == -1: return Directions.SOUTH
        return Directions.STOP
    
    def choose_action(self, game_state):

        actions = game_state.get_legal_actions(self.index)
        
        if None in actions:
            actions.remove(None)  # to avoid the ilegal action, remove the None in the actions list

        # If there is no legal movement in the list, we choose random one
        if len(actions) == 0:
            return random.choice(actions)

        food_list = self.get_food(game_state).as_list()
        my_pos = game_state.get_agent_position(self.index)
        nearest_food = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        # check how many food we have eaten
        if len(food_list) < self.previous_food_count:

            self.food_eaten += 1  # update the number of the food we got

            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)] #get the enemy status
            ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None] #get the enemy (as ghost) position
            scared_ghosts = [g for g in ghosts if g.scared_timer > 0] #if there is any scared ghost (enemy) in the game 

            scared_time_remaining = min([g.scared_timer for g in scared_ghosts], default=0) #the scare time left

            #once we got 3 dots and there is no scared ghost enemy in the game, we go back to our defense area(here we use the location of self.start) to store the score
            #if there is any scared ghost enemy, we keep eating the food until the timer (step) are less then 10.
            if self.food_eaten >= 3 and (not scared_ghosts or scared_time_remaining <= 10):
                # once the state change to defense, we cancel this movement, since the score is already been stored.
                if game_state.get_agent_state(self.index).is_pacman == False:
                    self.food_eaten = 0  # reset the food counter
                    self.previous_food_count = len(food_list)  # since we have eaten some foods, reset the total food count in the game

                    path = self.a_star_search(game_state, my_pos, nearest_food)
                    if path and len(path) > 1:
                        next_pos = path[1]
                        action = self.direction_between(my_pos, next_pos)
                        if action in actions:
                            return action

                    return random.choice(actions)

                if not actions:
                    return random.choice(actions)

                # use a* to get the min distance of from self.start to current location
                path = self.a_star_search(game_state, my_pos, self.start)

                if len(path) > 1:
                    next_pos = path[1]
                    best_action = self.direction_between(my_pos, next_pos)

                    #make sure the actions are legal
                    if best_action in actions:
                        return best_action

                # if a* failed
                return random.choice(actions)

        food_list = self.get_food(game_state).as_list()
        my_pos = game_state.get_agent_position(self.index)
        nearest_food = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        self.previous_food_count = len(food_list)

        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return random.choice(actions)

        # Generate candidate actions using A*
        a_star_actions = []
        if food_list:
            # Use A* to find path from current position to nearest food
            path = self.a_star_search(game_state, my_pos, nearest_food)
            if path and len(path) > 1:
                next_pos = path[1]  # Get next position in path
                action = self.direction_between(my_pos, next_pos)  # Convert to action
                if action in actions:
                    a_star_actions.append(action)  # Add valid action to list

        # Merge candidate actions and evaluate
        # Combine A* actions with other possible actions, remove duplicates
        candidate_actions = list(set(a_star_actions + actions))  
        candidate_actions = [a for a in candidate_actions if a is not None]  # Filter out None
        # Evaluate each candidate action
        values = [self.evaluate(game_state, a) for a in candidate_actions]  
        max_value = max(values)  # Find best evaluation value
        # Select actions with best evaluation value
        best_actions = [a for a, v in zip(candidate_actions, values) if v == max_value]  

        # to avoid the problem that stuck in the corner
        if game_state.get_agent_state(self.index).is_pacman == True:
            # once there are only 2 actions available (may cause the stuck problem), we mandatory choose the first one for next action
            if len(actions) == 2:
                valid_actions = [action for action in actions if action is not None]
                
                if valid_actions:
                    best_action = valid_actions[0]
                    return best_action
                else:
                    # if all the actions are ilegal, we choose random one
                    best_action = random.choice(actions)
                    return best_action

            # the normal offensive mode actions
            a_star_best_actions = [a for a in best_actions if a in a_star_actions]
            if a_star_best_actions:
                return random.choice(a_star_best_actions)
            
        a_star_best_actions = [a for a in best_actions if a in a_star_actions]
        if a_star_best_actions:
            return random.choice(a_star_best_actions)

        # If no a* actions use regular random choice
        if best_actions:
            return random.choice(best_actions)

        # If no valid actions, choose random from legal actions
        return random.choice(actions)


    def get_successor(self, game_state, action):
        if action is None:
            return game_state

        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()

        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor


    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        if action is None:
            return features

        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # count the invaders' number
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features


    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class Offensive(RunningPacmanCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # To maximize the score (eating food)

        # Compute distance to the nearest food
        my_pos = successor.get_agent_state(self.index).get_position()
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Get enemy ghost states
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        # Separate normal ghosts and scared ghosts
        scared_ghosts = [g for g in ghosts if g.scared_timer > 0]  # Scared ghosts
        normal_ghosts = [g for g in ghosts if g.scared_timer == 0]  # Dangerous ghosts

        # Avoid only normal ghosts (ignore scared ghosts)
        if len(normal_ghosts) > 0:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in normal_ghosts]
            features['ghost_penalty'] = max(0, min(ghost_dists))  # Avoid only normal ghosts

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 200,       # Maximizing score by eating food
            'distance_to_food': -5,       # Get closer to food
            'ghost_penalty': 1500         # Penalize getting close to normal (non-scared) ghosts
        }





class Defensive(RunningPacmanCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. 
    If it becomes a scared ghost, it avoids invaders instead of chasing them.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman: # if status is pacman
            features['on_defense'] = 0

        # count the invaders number
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            closest_invader_dist = min(dists) # calculate the closest dist to the invader

            if my_state.scared_timer > 0:
                # if my status is scared ghosts, avoid the invaders
                features['avoid_invaders_distance'] = sqrt(closest_invader_dist)
            else:
                features['invader_distance'] = closest_invader_dist

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        weights = {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2
        }

        
        my_state = game_state.get_agent_state(self.index)
        if my_state.scared_timer > 0:
            weights['avoid_invaders_distance'] = 4  # weight to avoid the invaders
            weights['invader_distance'] = 0  # set to 0 to not close to the invader
        

        return weights