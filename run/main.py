import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import os
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from search_algorithms.pareto_nrpa.pareto_nrpa import ParetoNRPA

from search_spaces import make_search_space

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

algorithms = {
    "pareto_nrpa": ParetoNRPA
}


if __name__ == "__main__":
    @hydra.main(config_path="../conf", config_name="config", version_base=None)
    def main(cfg: DictConfig):
        # Save config to tensorboard log dir
        print("Configuration:\n", OmegaConf.to_yaml(cfg))
        cfg_path = os.path.join(get_original_cwd(), cfg.log_dir, "config.yaml")
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        OmegaConf.save(cfg, cfg_path)

        problem = make_search_space(cfg)
        algorithm = algorithms[cfg.search.algorithm]
        agent = algorithm(cfg, problem)
        results = agent.main()

        df = pd.DataFrame(columns=["sequence", "W", "P"].extend([f"F_{i}" for i in range(problem.n_objectives)]))
        for res in results:
            row = {"sequence": res.get("X"), "W": res.get("W"), "P": res.get("P")}
            for i, f in enumerate(res.get("F")):
                row[f"F_{i}"] = f
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            
        # Use the run name from hydra config
        run_name = cfg.run_name if "run_name" in cfg else "results"
        out_path = os.path.join(get_original_cwd(), cfg.log_dir, f"{run_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"Results exported to {out_path}")
    main()
