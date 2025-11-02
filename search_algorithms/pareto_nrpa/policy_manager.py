import copy
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
import numpy as np


class PolicyManager:

    def __init__(self, alpha):
        self.policies = {}
        self.weights = {}
        self.alpha = alpha

    @property
    def n_policies(self):
        return len(self.policies.keys())

    def get_policy(self, index):
        assert index in self.policies.keys(), f"{index} is not one of the policies"
        return self.policies[index]

    def update_policy(self, index, pol):
        self.policies[index] = pol

    def delete_policy(self, index):
        assert index in self.policies.keys(), f"{index} is not one of the policies"
        self.policies.pop(index)

    def update_weights(self, optimal_set):
        """
        Updating the weights for the policies. The weight depends on the number of points on the pareto front and on the crowding distance of such points.
        """
        crowding = RankAndCrowding()
        distances = crowding.do(problem=Problem(n_constr=0), pop=optimal_set)
        dist = np.where(distances.get("crowding") == np.inf, 1, distances.get("crowding"))
        
        pol_w = {k: 1 for k in self.policies.keys()}
        counts = pol_w.copy()
        
        for p, c in zip(optimal_set.get("P"), dist):
            counts[p] += 1
        weights = {k: v/counts[k] for k, v in pol_w.items()}
        for pol in self.policies.keys():
            self.weights[pol] = weights[pol]
            
    def copy(self):
        pm = self.__class__(alpha=self.alpha)
        pm.weights = self.weights.copy()
        for k, v in self.policies.items():
            pm.update_policy(k, v.copy())
        return pm

    def adapt(self, optimal_set, algorithm, level=0):
        """
        Adapt policies based on the optimal set.
        """
        crowding = RankAndCrowding()
        distances = crowding.do(problem=Problem(n_constr=0), pop=optimal_set)
        dist = np.where(distances.get("crowding") == np.inf, 2, distances.get("crowding"))

        for elem, dis in zip(optimal_set, dist):
            # if level >= algorithm.level:
            #     print(f"[LEVEL {level}] Adapting towards state: {elem.get('X')} with score {elem.get('F')}")
            policy_index = elem.get("P")
            sequence = elem.get("X")
            policy = self.policies[policy_index]

            node_type = algorithm.node_type
            node = node_type(algorithm.problem)
            pol_prime = policy.copy()

            for i, action in enumerate(sequence[1:]):
                best_code = algorithm.problem._move_coder(node.state, action)
                pol_prime[best_code] = pol_prime.get(best_code, 0) + (self.alpha*dis)
                z = 0
                o = {}
                available_moves = node.get_actions_tuples()
                move_codes = [algorithm.problem._move_coder(node.state, m) for m in available_moves]

                for move, move_code in zip(available_moves, move_codes):
                    o[move_code] = np.exp(policy.get(move_code, 0))
                    z += o[move_code]
                for move, move_code in zip(available_moves, move_codes):
                    pol_prime[move_code] = pol_prime.get(move_code, 0) -  (self.alpha*dis) * (o[move_code] / z)

                node.play_action(action)
            self.update_policy(policy_index, pol_prime)
        self.update_weights(optimal_set)
