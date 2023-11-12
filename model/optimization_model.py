from .pyomo_model import PartialModel
from itertools import combinations
from typing import Optional
from pyomo.environ import value
import pyomo.environ as pyo
import numpy as np
import pandas as pd
from tqdm import tqdm


class OptimizationModel:
    """
    Optimization model for a given product.

    Args:
        data (pd.DataFrame): data for a given product
        beta (float): percentage of demand to be satisfied
        Pmax (int): number of months to produce
        Cmax (int): maximum inventory
        C (float): cost of inventory per unit (per month)

    Attributes:
        model_list (list): list of possible models
        data (pd.DataFrame): data for a given product
        beta (float): percentage of demand to be satisfied
        Pmax (int): maximum number of months to produce
        Cmax (int): maximum inventory
        C (float): cost of inventory per unit (per month)
    """
    def __init__(
        self,
        data: pd.DataFrame,
        beta: float,
        Pmax: int,
        Cmax: int,
        C: float,
    ):
        self.model_list = []
        self.data = data
        self.beta = beta
        self.Pmax = Pmax
        self.Cmax = Cmax
        self.C = C
        self.build_models()

    def build_models(self) -> None:
        """
        Build all the possible models for a given Pmax.

        Returns:
            None
        """
        COMBINATIONS = np.array(
            [
                [1 if i in comb else 0 for i in range(12)]
                for comb in combinations(np.arange(12), self.Pmax)
            ]
        )
        self.model_list = [
            PartialModel(
                self.data,
                self.beta,
                self.Cmax,
                self.C,
                delta=comb,
            )
            for comb in COMBINATIONS
        ]

    def solve(self, solver_path: str) -> Optional[PartialModel]:
        """
        Solve the model over all the possible combinations of delta. Return the
        optimal model if a feasible solution is found, None otherwise.

        Args:
            solver_path (str): path to the solver

        Returns:
            Optional[PartialModel]: optimal model if a feasible solution is
            found, None otherwise
        """
        feaseable_models = []
        with tqdm(total=len(self.model_list)) as pbar:
            for i, model in enumerate(self.model_list):
                opt = pyo.SolverFactory("glpk", executable=solver_path)
                result = opt.solve(model)
                if (result.solver.status == pyo.SolverStatus.ok and 
                    result.solver.termination_condition == pyo.TerminationCondition.optimal
                ):
                    feaseable_models.append(model)
                pbar.update(1)

        if feaseable_models:
            return self.get_optimal_model(feaseable_models)

    @staticmethod
    def get_optimal_model(feaseable_models: list) -> PartialModel:
        """
        Get the optimal model from a list of feaseable models.
        
        Args:
            feaseable_models (list): list of feaseable models
            
        Returns:
            PartialModel: optimal model
        """
        index_best_model = np.argmin(
            [value(model.objective()) for model in feaseable_models]
        )
        return feaseable_models[index_best_model]
    
