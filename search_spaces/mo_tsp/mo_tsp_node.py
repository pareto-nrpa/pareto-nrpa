
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from search_spaces import Node
from .mo_tsp import MO_TSP


class MO_TSPNode(Node):

    def __init__(self, problem:MO_TSP, path=None, visited=None):
        self.n_cities = problem.n_cities
        self.path = path if path is not None else [0]
        self.visited = visited.copy() if visited is not None else {0}

    @property
    def is_terminal(self):
        return self.visited == set(range(self.n_cities))
    
    @property
    def state(self):
        return self.path
    
    def get_actions_tuples(self):
        return [int(i) for i in range(self.n_cities) if i not in self.visited]
    
    def play_path(self, path): 
        """
        Play a complete path by visiting all cities in the path.
        :param path: List of city indices representing the path to play.
        """
        assert path[0] == 0, "Path must start at the depot (city 0)."
        for j in path[1:]:
            if j in self.visited:
                raise ValueError(f"City {j} has already been visited.")
            self.play_action(j)

    def play_action(self, action):
        if action in self.visited:
            raise ValueError(f"City {action} has already been visited.")
        self.path.append(action)
        self.visited.add(action)

    def playout(self, policy, move_coder, softmax_temp=1.0):
        while not self.is_terminal:
            available_actions = self.get_actions_tuples()
            policy_values = [policy.get(move_coder(self.path, act), 0) for act in available_actions]
            exp_values = np.exp(np.array(policy_values) / softmax_temp)
            probs = exp_values / np.sum(exp_values)
            action_index = random.choices(np.arange(len(available_actions)), weights=probs)[0]
            self.play_action(available_actions[action_index])
        return self.path

    def random_playout(self):
        while not self.is_terminal:
            available_actions = self.get_actions_tuples()
            if len(available_actions) == 0:
                break
            action = np.random.choice(available_actions)
            self.play_action(action)
        return self.path

if __name__ == "__main__":
    problem = MO_TSP(n_cities=24, n_objectives=10, seed=42)

    for i in range(10):

        mynode = MO_TSPNode(problem)
        while not mynode.is_terminal():
            actions = mynode.get_actions_tuples()
            action = np.random.choice(actions)  # Select the first available action
            mynode.play_action(action)
            
        print(f"Path: {mynode.path}, Visited: {mynode.visited}")
        print(f"Objective values: {problem._evaluate_path(mynode.path)}")