# Pareto-NRPA: A Novel Monte-Carlo Search Algorithm for Multi-Objective Optimization

Welcome to the repository for the **Pareto-NRPA** algorithm, which was published in the 28th European Conference on Artificial Intelligence (ECAI 2025).

The paper is available here: [https://arxiv.org/abs/2507.19109](https://arxiv.org/abs/2507.19109)

## Adding your own problem to Pareto-NRPA

### Creating a search space object

1. In the `search_spaces` folder, create a new folder `problem` for your use case.

2. In this folder, create a `problem.py` file. The `problem.py` object should instanciate the following attributes and methods:
    - (_attribute_) `n_objectives`
    - (_method_) `_move_coder`: Assigns a unique code for a (state, action) couple. The way to assign this value (e.g. using Zobrist hashing) is up to the user.
    - (_method_) `_evaluate(self, node)`: takes as input a solution node and returns the objective function values for this node

3. Create a `problem_node.py` file. This file represents the actual object that is manipulated during the search. This object should instanciate the following attributes and methods:
    - (_property method_) `is_terminal`: True if the node has reached a terminal state
    - (_property method_) `state`: Returns the current state
    - (_method_) `playout(self, policy, _move_coder, softmax_temperature)`: Performs a playout using a policy $\pi$

4. In the file `search_algorithms/pareto_nrpa.py`: 
    - Add your node type to the `_adapt_search_space` method.

### Creating a configuration file

Pareto-NRPA uses Hydra to manage configurations for problems and hyperparameters. 

If you want to add support for different problem configurations, you can add attributes to a `problem.yaml` file which is used by the search space constructor.

## Running a Pareto-NRPA search

### In a terminal

1. Run the `run/main.py` file in the terminal. Using hydra, you can change the configuration on-the-fly using command line-level parameters. 

    For instance, you can call `python main.py problem=mo_tsptw search.n_iter=20` to run a Pareto-NRPA search with 20 iterations per level on the TSPTW problem.

### Using python

1. Instanciate the Hydra configuration using `@hydra.main(config_path="../conf", config_name="config", version_base=None)`
2. You can define a new `.yaml` configuration file or edit the existing one to include your problem.
3. Instanciate your problem: `problem = make_search_space(cfg)`
4. Intanciate Pareto-NRPA: `algorithm = ParetoNRPA(cfg, problem)`
5. Run the search using the `main()` method of Pareto-NRPA. The `main()` method outputs a PyMoo population file containing non-dominated solutions and their addociated objective function values.

## Cite Pareto-NRPA

If you use Pareto-NRPA in a research publication, please consider citing the original paper:

```
@incollection{lallouet_pareto-nrpa_2025,
	title = {Pareto-{NRPA}: {A} {Novel} {Monte}-{Carlo} {Search} {Algorithm} for {Multi}-{Objective} {Optimization}},
	shorttitle = {Pareto-{NRPA}},
	url = {https://ebooks.iospress.nl/doi/10.3233/FAIA251394},
	language = {en},
	urldate = {2025-10-29},
	booktitle = {{ECAI} 2025},
	publisher = {IOS Press},
	author = {Lallouet, Noé and Cazenave, Tristan and Enderli, Cyrille},
	year = {2025},
	doi = {10.3233/FAIA251394},
	pages = {4848--4855},
}
```
## Authors

- Noé Lallouet
- Tristan Cazenave
- Cyrille Enderli
