import copy
import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

class MO_TSP:

    def __init__(self, config):
        self.n_cities = config.problem.n_cities
        self.n_objectives = config.problem.n_objectives
        self.seed = config.seed
        self.coordinates = np.empty((self.n_objectives, self.n_cities, 2))
        self._generate_coordinates()
        self.distance_matrices = self._compute_distance_matrices()
        self.nadir = 0.52*self.n_cities  # Nadir point approximately corresponding to a random path.
        self._move_coder = lambda path, move: path[-1] * self.n_cities + move

    def _generate_coordinates(self):
        rng = np.random.default_rng(self.seed)
        for obj in range(self.n_objectives):
            self.coordinates[obj] = rng.random((self.n_cities, 2))

    def _compute_distance_matrices(self):
        """
        Compute distance matrices for each objective.
        Returns:
        A numpy array of shape (n_objectives, n_cities, n_cities) containing the distance matrices.
        """
        distance_matrices = np.empty((self.n_objectives, self.n_cities, self.n_cities))
        for obj in range(self.n_objectives):
            coords = self.coordinates[obj]
            distance_matrices[obj] = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
        return distance_matrices
    
    def _evaluate(self, node):
        """
        Evaluate the total distance for each objective given a path.
        Args:
            path: A list of city indices representing the order of visitation.
        Returns:
            A numpy array of shape (n_objectives,) containing the total distances for each objective.
        """
        if hasattr(node, 'path'):
            path = copy.deepcopy(node.path)
        elif isinstance(node, list):
            path = copy.deepcopy(node)
        else:
            raise ValueError("Input must be a list of city indices or a node with a 'path' attribute.")
        path.append(path[0])  # return to starting city
        total_distances = np.empty(self.n_objectives)
        for obj in range(self.n_objectives):
            dist_matrix = self.distance_matrices[obj]
            total_distances[obj] = sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
        return total_distances
    
    def _evaluate_partial(self, node):
        """
        Evaluate the partial distance for each objective given a partial path.
        Args:
            node: A node containing a partial path or a list of city indices.
        """
        if hasattr(node, 'path'):
            path = copy.deepcopy(node.path)
        elif isinstance(node, list):
            path = copy.deepcopy(node)
        else:
            raise ValueError("Input must be a list of city indices or a node with a 'path' attribute.")
        total_distances = np.empty(self.n_objectives)
        for obj in range(self.n_objectives):
            dist_matrix = self.distance_matrices[obj]
            total_distances[obj] = sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
        return total_distances
    
    def visualize(self, x, fig=None, ax=None, show=True, label=True):
        
        
        with plt.style.context('ggplot'):

            if fig is None or ax is None:
                fig, ax = plt.subplots(1, self.n_objectives, figsize=(6*self.n_objectives, 5))
                if self.n_objectives == 1:
                    ax = [ax]
                for n, color in zip(range(self.n_objectives), ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]):
                    ax[n].set_title(f"Objective {n+1} | Path score: {self._evaluate(x)[n]:.4f}")
                    ax[n].set_xlim(0, 1)
                    ax[n].set_ylim(0, 1)
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
        problem = MO_TSP(cfg)
        problem.visualize(list(range(20)))
    main()
    