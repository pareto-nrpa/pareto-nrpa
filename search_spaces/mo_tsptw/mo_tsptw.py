import copy
import os
import pickle
import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import pandas as pd
import re
import io
from pathlib import Path

class MO_TSPTW:

    def __init__(self, config):
        module_root = Path(__file__).resolve().parent
        self.data_path = module_root / "data"
        self.instance = config.problem.instance
        self.n_objectives = config.problem.n_objectives  # Can set 1 for single objective TSPTW
        self.coordinates = self._compute_coordinates()
        self.distance_matrices = self._compute_distance_matrices()
        self._move_coder = lambda path, move: path[-1] * self.n_cities + move


    def parse_file(self, kind="euclidean"):
        if kind == "euclidean":
            with open(os.path.join(self.data_path, "SolomonTSPTW", f"{self.instance}.txt"), 'r', encoding="utf-8") as f:
                s = f.read()
            n_cities, text = s.split("\n", 1)
            new_text = ""
            for line in text.split("\n"):
                new_text += re.sub("\s+", ";", line.strip()) + "\n"
            n_cities = int(n_cities)
            df = pd.read_csv(io.StringIO(new_text), sep=';', lineterminator='\n', header=None)
            data = {}
            matrix = np.zeros((n_cities, n_cities))
            for i in range(n_cities):
                for j in range(n_cities):
                    matrix[i, j] = np.sqrt((df.iloc[i, 1] - df.iloc[j, 1]) ** 2 + (df.iloc[i, 2] - df.iloc[j, 2]) ** 2) + df.iloc[i, 6]
            # print(matrix)
            for i in range(n_cities):
                data[i] = {'time_window': (int(df.iloc[i, 4]), int(df.iloc[i, 5])), 'service_time': int(0)}
            return matrix, data

        with open(self.data_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        n_cities = int(lines[0])
        data = {}
        matrix = np.zeros((n_cities, n_cities))
        for k in range(n_cities):
            l = lines[k+1].rstrip()
            for i, val in enumerate(l.split(" ")):
                matrix[k, i] = float(val)
        for j in range(n_cities):
            l = lines[j+n_cities+1].rstrip()
            for i, char in enumerate(l):
                if char == " ":
                    entry = l[:i]
                    break
            l_reverse = l[::-1]
            print(l_reverse)
            for i, char in enumerate(l_reverse):
                if char == " ":
                    out = l_reverse[:i][::-1]
                    break
            print(f"City {j}: {entry}, {out}")
            data[j] = {'time_window': (int(entry), int(out)), 'service_time': int(0)}
        return matrix, data

    def _compute_coordinates(self):
        with open(os.path.join(self.data_path, "SecondaryCost", f"coordinates_{self.instance}.pickle"), "rb") as f:
            secondary_coords = pickle.load(f)

        with open(os.path.join(self.data_path, "SolomonTSPTW", f"{self.instance}.txt"), 'r', encoding="utf-8") as f:
            s = f.read()
        n_cities, text = s.split("\n", 1)
        new_text = ""
        for line in text.split("\n"):
            new_text += re.sub("\s+", ";", line.strip()) + "\n"
        n_cities = int(n_cities)
        df = pd.read_csv(io.StringIO(new_text), sep=';', lineterminator='\n', header=None)

        # Build dictionary of (x, y) coordinates from the primary objective file
        primary_coords = {
            city_idx: (float(df.iloc[city_idx, 1]), float(df.iloc[city_idx, 2]))
            for city_idx in range(n_cities)
        }

        # Ensure secondary coordinates follow the same float format
        secondary_coords = {
            int(city_idx): (float(coord[0]), float(coord[1]))
            for city_idx, coord in secondary_coords.items()
        }

        if self.n_objectives == 2:
            # Store both dictionaries for reference and construct array used by visualisation
            self.city_coordinates = {
                0: primary_coords,
                1: secondary_coords,
            }

            coords_array = np.zeros((self.n_objectives, n_cities, 2), dtype=float)
            for city in range(n_cities):
                coords_array[0, city] = primary_coords[city]
                coords_array[1, city] = secondary_coords.get(city, primary_coords[city])
        elif self.n_objectives == 1:
            self.city_coordinates = {
                0: primary_coords,
            }
            coords_array = np.zeros((1, n_cities, 2), dtype=float)
            for city in range(n_cities):
                coords_array[0, city] = primary_coords[city]
        else:
            raise ValueError("n_objectives must be 1 or 2 for MO_TSPTW.")

        return coords_array


    def _compute_distance_matrices(self):
        """
        Compute distance matrices for each objective.
        Returns:
        A numpy array of shape (n_objectives, n_cities, n_cities) containing the distance matrices.
        """
        primary, data = self.parse_file()
        if self.n_objectives == 1:
            distance_matrices = np.expand_dims(primary, axis=0)
            self.cities_data = data
            self.n_cities = int(primary.shape[0])
            return distance_matrices
        
        secondary = np.load(os.path.join(self.data_path, "SecondaryCost", f"{self.instance}.npy"))
        distance_matrices = np.stack([primary, secondary], axis=0)
        self.cities_data = data
        self.n_cities = int(primary.shape[0])
        return distance_matrices

    def _evaluate(self, node_):
        """
        Calculate the reward for the current state. Assumes the goal is to minimize total travel time.

        :return: Negative of the total travel time if the state is complete, otherwise 0.
        """
        assert node_.is_terminal, "Path is not complete."
        node = copy.deepcopy(node_)
        node.path.append(node.path[0])  # return to starting city
        total_distances = np.empty(self.n_objectives)

        n_violations = 0
        score = 0
        for i in range(len(node.path) - 1):
            from_city = node.path[i]
            to_city = node.path[i+1]
            score -= self.distance_matrices[0][from_city][to_city]
        # print(f"Score: {score}, current time: {self.current_time}")
        for i, (city, time) in enumerate(node.visit_times):
            if time > self.cities_data[city]['time_window'][1]:
                # print(f"Time window violation at city {city} (index {i} in path): arrived at {time}, latest allowed {self.cities_data[city]['time_window'][1]}")
                n_violations +=1

        if node.n_objectives == 1:
            total_distances[0] = -score + 1e6*n_violations
            return total_distances

        secondary_score = 0
        for i in range(len(node.path) - 1):
            from_city = node.path[i]
            to_city = node.path[i+1]
            secondary_score -= self.distance_matrices[1][from_city][to_city]
        total_distances[0] = -score + 1e6*n_violations
        total_distances[1] = -secondary_score + 1e6*n_violations
        return total_distances

    def visualize(self, x, fig=None, ax=None, show=True, label=True):

        with plt.style.context('ggplot'):

            if fig is None or ax is None:
                fig, ax = plt.subplots(1, self.n_objectives, figsize=(6*self.n_objectives, 5))
                if self.n_objectives == 1:
                    ax = [ax]
                for n, color in zip(range(self.n_objectives), ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]):
                    ax[n].set_title(f"Objective {n+1}")
                    for city in range(self.n_cities):
                        ax[n].scatter(self.coordinates[n][city][0], self.coordinates[n][city][1], s=25, c=color)
                        if label:
                            ax[n].annotate(str(city), xy=(self.coordinates[n][city][0]+0.01, self.coordinates[n][city][1]+0.01), fontsize=10, ha="center", va="center", color="black")
                    # Plot the path
                    for i in range(len(x)):
                        current = x[i]
                        next_ = x[(i + 1) % len(x)]
                        ax[n].plot((self.coordinates[n][current][0], self.coordinates[n][next_][0]), (self.coordinates[n][current][1], self.coordinates[n][next_][1]), linestyle='--', c=color)

            if show:
                plt.show()
            else:
                return fig, ax

if __name__ == "__main__":
    @hydra.main(config_path="../../conf", config_name="config", version_base=None)
    def main(cfg: DictConfig):
        problem = MO_TSPTW(cfg)
        problem._compute_distance_matrices()
        print(problem.city_coordinates)
        problem.visualize(list(np.arange(problem.n_cities)))
    main()
    
