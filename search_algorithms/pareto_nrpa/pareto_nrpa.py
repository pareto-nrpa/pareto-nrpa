import time
import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from search_algorithms.pareto_nrpa.policy_manager import PolicyManager
from search_spaces.mo_tsp.mo_tsp_node import MO_TSPNode
from search_spaces.mo_tsptw.mo_tsptw_node import MO_TSPTWNode
from tqdm import tqdm


class ParetoNRPA:

    def __init__(self, config, problem):

        self.config = config
        self.problem = problem
        self.level = config.search.level
        self.n_iter = config.search.n_iter
        self.alpha = config.search.alpha
        self.advancement = 0
        self.limited_repetitions = config.search.limited_repetitions
        self.best_score = np.inf
        self.pm = PolicyManager(alpha=self.alpha)
        self.n_policies = config.search.n_policies

        if self.config.search.progressbar:
            total_steps = self.n_iter ** self.level
            self.progress_bar = tqdm(total=total_steps, desc="Progress", unit="step")


        self._adapt_search_space()
        self._initialize()

    def _adapt_search_space(self):
        if self.config.problem.name == "mo_tsp":
            self.node_type = MO_TSPNode

        elif self.config.problem.name == "mo_tsptw":
            self.node_type = MO_TSPTWNode

        ## Custom problems can be added here

        else:
            raise ValueError(f"Problem {self.config.problem.name} has not been implemented for Pareto NRPA.")

    def _initialize(self):
        # Set initial policy
        for i in range(self.n_policies):
            self.pm.update_policy(i, {})
            self.pm.weights[i] = 1

    def _pareto_nrpa(self, level, policy_manager):

        if level == 0:
            # Choose a random policy
            policy_index = np.random.choice(list(self.pm.policies.keys()))
            node = self.node_type(self.problem)
            sequence = node.playout(policy=policy_manager.get_policy(policy_index), 
                                move_coder=self.problem._move_coder, 
                                softmax_temp=self.config.search.softmax_temperature)
            reward = self.problem._evaluate(node)
            #PyMoo individual creation
            new_individual = Individual()
            new_individual.set("X", node.state)
            new_individual.set("F", reward)
            new_individual.set("P", policy_index)
            self.advancement += 1
            if self.config.search.progressbar: self.progress_bar.update(1)

            return Population.merge(Population(), new_individual)

        
        else:
            optimal_set = Population()
            count_threshold = 0  # NRPA with limited repetitions

            for i in range(self.n_iter):
                # if level >= 2: print(f"Level {level} Iteration {i+1}/{self.n_iter}, Optimal set size: {len(optimal_set)}")

                if level == 1:
                    result = self._pareto_nrpa(level - 1, policy_manager)
                else:
                    # Copy policy manager to avoid modifying the original one
                    pm_copy = policy_manager.copy()
                    result = self._pareto_nrpa(level - 1, pm_copy)

                # Limited repetitions check
                if self.limited_repetitions > 0:
                    
                    # Check for improvement
                    result_paths = set(tuple(ind.get("X")) for ind in result)
                    optimal_paths = set(tuple(ind.get("X")) for ind in optimal_set)
                    
                    sequences_identical = result_paths.issubset(optimal_paths)
                    if sequences_identical: 
                        count_threshold += 1
                    else:
                        count_threshold = 0

                    if count_threshold >= self.limited_repetitions:
                        # if level >= 1: print(f"[LEVEL {level}] No improvement in {self.limited_repetitions} iterations, stopping early @ iter {i+1}.")
                        break
                
                optimal_set = Population.merge(optimal_set, result)

                # Non-dominated sorting to get the Pareto front
                nds = NonDominatedSorting()
                fronts = nds.do(optimal_set.get("F"))
                indexes = []
                for f in fronts[0]:
                    el = optimal_set[f]
                    duplicate = False
                    for dimension in range(len(el.get("F"))):
                        if el.get("F")[dimension] in [optimal_set[e].get("F")[dimension] for e in indexes]:
                            duplicate = True
                            break
                    if not duplicate:
                        indexes.append(f)
                
                # Modifying the optimal set so that each policy is represented
                osi = optimal_set[indexes]  # Optimal set incomplete
                for p in policy_manager.policies.keys():
                    if len(osi[osi.get("P") == p]) == 0:
                        # print(f"Policy {p} has no point in the pareto set.")
                        finished = False
                        j = 1
                        while not finished:
                            if j  == len(fronts):
                                break
                            for f in fronts[j]:
                                if optimal_set[f].get("P") == p:
                                    # print(f"Adding element from front {j}")
                                    indexes.append(f)
                                    finished = True
                                    break
                            j += 1

                optimal_set = optimal_set[indexes]  # Optimal set complete

                # Pareto Adapt
                t0 = time.time()
                policy_manager.adapt(optimal_set, self, level=level)
                t1 = time.time()
                # print(f"[LEVEL {level}] Pareto ADAPT Time taken: {t1 - t0:.4f}s")
            return optimal_set

    def main(self):
        
        print(f"Running Neural Pareto-NRPA on problem {self.config.problem.name}")
        results = self._pareto_nrpa(self.level, self.pm)

        if self.config.search.progressbar:
            self.progress_bar.close()

        print(f"Objective values: {results.get('F')}")
        print(f"Total steps: {self.advancement}")
        return results