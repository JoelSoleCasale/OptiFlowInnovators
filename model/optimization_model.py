from pyomo_model import PartialModel
from itertools import combinations
from pyomo.environ import value
import pyomo.environ as pyo
import numpy as np
import pandas as pd


class OptimizationModel:
    def __init__(
        self,
        data: pd.DataFrame,
        beta: float,
        Pmax: int,
        Cmax: int,
        C: float,
        gamma: float = 1,
    ):
        self.model_list = []
        self.data = data
        self.beta = beta
        self.Pmax = Pmax
        self.Cmax = Cmax
        self.gamma = gamma
        self.C = C
        self.build_models()

    def build_models(self):
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
                self.Pmax,
                self.Cmax,
                self.C,
                gamma=self.gamma,
                delta=comb,
            )
            for comb in COMBINATIONS
        ]

    def solve(self, solver_path):
        for model in self.model_list:
            opt = pyo.SolverFactory("glpk", executable=solver_path)
            opt.solve(model)

        model = self.get_optimal_model()
        return model

    def get_optimal_model(self):
        return self.model_list[
            np.argmin([value(model.objective()) for model in self.model_list])
        ]
