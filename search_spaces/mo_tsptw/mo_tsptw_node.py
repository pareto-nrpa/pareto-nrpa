from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from search_spaces import Node
from search_spaces.mo_tsptw.mo_tsptw import MO_TSPTW

class MO_TSPTWNode(Node):

    def __init__(self, problem: MO_TSPTW, path=None, current_time=0, visited=None, visit_times=[(0,0)]):
        self.n_cities = problem.n_cities
        self.cities_data = problem.cities_data
        self.distance_matrices = problem.distance_matrices  # primary objective
        self.n_objectives = problem.n_objectives
        self.path = path if path is not None else [0]
        self.visited = visited.copy() if visited is not None else {0}
        self.current_time = current_time  # time at current city
        self.visit_times = visit_times.copy()

    @property
    def is_terminal(self):
        required_cities = set(self.cities_data.keys())
        visited_in_path = set(self.path)
        return visited_in_path == required_cities
    
    def get_actions_tuples(self):
        """
        Generate valid actions (next cities to visit) from the current state.

        :return: List of valid actions as tuples. Each action is a tuple containing the next city ID.
        """
        actions = [None for _ in range(self.distance_matrices[0].shape[0])]

        nb = 0
        current_city = self.path[-1]
        required_cities = set(self.cities_data.keys()) - {0}
        unvisited = required_cities - self.visited
        # Check if all required cities are visited and the last city is not the depot
        if not unvisited and current_city != 0:
            return [0]

        for i in range(self.distance_matrices[0].shape[0]):
            if i in unvisited:
                actions[nb] = i
                if self.current_time + self.cities_data[current_city]['service_time'] + \
                        self.distance_matrices[0][current_city][i] > self.cities_data[i]['time_window'][1]:
                    nb += 1
        if nb == 0:
            for i in range(self.distance_matrices[0].shape[0]):
                if i in unvisited:
                    actions[nb] = i
                    trop_tard = False
                    for j in range(1, self.distance_matrices[0].shape[0]):
                        if j != i:
                            if j in unvisited:
                                if (
                                        (self.current_time <= self.cities_data[j]['time_window'][1]) and
                                        (self.current_time + self.distance_matrices[0][current_city][j] +
                                        self.cities_data[current_city]['service_time'] <=
                                        self.cities_data[j]['time_window'][1]) and
                                        (max(self.current_time + self.distance_matrices[0][current_city][i] +
                                            self.cities_data[i]['service_time'],
                                            self.cities_data[i]['time_window'][0]) >
                                        self.cities_data[j]['time_window'][1])
                                ):
                                    trop_tard = True
                                    break
                    if not trop_tard:
                        nb += 1
        if nb == 0:
            # print(f"Len actions is 0")
            for v in range(self.distance_matrices[0].shape[0]):
                if v in unvisited:
                    actions[nb] = v
                    nb += 1
        return actions[:nb]
    
    def play_path(self, path):
        """
        Play a complete path by visiting all cities in the path.
        :param path: List of city indices representing the path to play.
        """
        assert path[0] == 0, "Path must start at the depot (city 0)."
        for city in path[1:]:
            self.play_action(city)

    def play_action(self, city):
        """
        Apply an action to the current state and return the new state.

        :param action: A tuple containing the next city ID to visit.
        :return: New TSPTWState instance after applying the action.
        """
        
        j = city
        new_path = self.path.copy()
        new_path.append(j)
        current_city = self.path[-1]
        e_i, l_i = self.cities_data[j]['time_window']
        s_i = self.cities_data[current_city]['service_time']
        departure_time_i = self.current_time + s_i
        travel_time = self.distance_matrices[0][current_city][j]
        arrival_time_j = departure_time_i + travel_time
        wait_time = max(0, e_i - arrival_time_j)
        arrival_time_j += wait_time
        new_visited = self.visited.copy()
        if j != 0:
            new_visited.add(j)
        self.visit_times.append((j, arrival_time_j))
        self.path = new_path
        self.visited = new_visited
        self.current_time = arrival_time_j


    def neural_playout(self, policy_network=None, softmax_temp=1.0):
        while not self.is_terminal:
            available_actions = self.get_actions_tuples()
            if len(available_actions) == 0:
                break
            if policy_network is not None:
                path_tensor = torch.tensor(self.path, dtype=torch.long).to(policy_network.device)
                mask = torch.zeros(self.n_cities, device=policy_network.device)
                mask[available_actions] = 1.  # Mark available actions with 1
                logits = policy_network(path_tensor, mask)
                probs = F.softmax(logits / softmax_temp, dim=-1).detach().cpu().numpy().squeeze()
                action = np.random.choice(self.n_cities, p=probs)
            self.play_action(action)
        return self.path
    
    def policy_playout(self, policy_table, move_coder, softmax_temp=1.0):
        while not self.is_terminal:
            available_actions = self.get_actions_tuples()
            policy_values = [policy_table.get(move_coder(self.path, act), 0) for act in available_actions]
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
    import hydra
    from omegaconf import DictConfig
    import sys
    from hydra.utils import get_original_cwd

    @hydra.main(config_path="../../conf", config_name="config", version_base=None)
    def main(cfg: DictConfig):
        sys.path.insert(0, get_original_cwd())
        problem = MO_TSPTW(cfg)
        node = MO_TSPTWNode(problem)
        liszt = [0, 33, 32, 31, 28, 21, 23, 20, 15, 18, 19, 16, 17, 22, 24, 41, 42, 40, 43, 12, 13, 14, 37, 29, 36, 8, 10, 11, 27, 9, 30, 38, 35, 39, 7, 26, 5, 3, 4, 2, 6, 25, 1, 45, 34, 44, 0]
        for city in liszt[1:]:
            print(f"Visiting city {city} at time {node.current_time}. City time window: {problem.cities_data[city]['time_window']}")
            node.play_action(city)
        y= problem._evaluate(node)
        print(y)
        problem.visualize(node)
    main()
