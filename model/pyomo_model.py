import pyomo.environ as pyo
import pandas as pd


class PartialModel(pyo.ConcreteModel):
    def __init__(
        self,
        data: pd.DataFrame,
        beta: float,
        Pmax: int,
        Cmax: int,
        C: float,
        gamma: float = 1,
        delta: list = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data = self.process_data(data)
        self.beta = beta
        self.Pmax = Pmax
        self.Cmax = Cmax
        self.gamma = gamma
        self.C = C
        self.delta = delta
        self.build_model()

    def build_model(self):
        self.set_indices()
        self.set_parameters()
        self.set_variables()
        self.set_constraints()
        self.set_objective()

    def set_indices(self):
        self.T = pyo.Set(initialize=range(1, 13), doc="Time periods")

    def set_variables(self):
        self.p = pyo.Var(
            self.T, domain=pyo.NonNegativeReals, doc="Production amount", initialize=0
        )

    def set_parameters(self):
        self.xi = self.compute_xi()
        self.v = self.compute_consumption_rate()
        
    def set_constraints(self):
        self.set_inventory_constraint()

    def set_inventory_constraint(self):
        self.set_nonegative_inventory_constraint()
        self.set_max_inventory_constraint()
        self.set_sufficient_inventory_constraint()

    def inventory(self, t):
        return (
            sum(self.delta[i-1] * self.p[i] - self.v[i] for i in range(1, t))
            + self.delta[t-1] * self.p[t]
        )

    def set_nonegative_inventory_constraint(self):
        def rule(model, t):
            return model.inventory(t) >= 0

        self.nonegative_inventory_constraint = pyo.Constraint(
            self.T, rule=rule, doc="Non-negative inventory constraint"
        )

    def set_max_inventory_constraint(self):
        def rule(model, t):
            return model.inventory(t) <= self.Cmax

        self.max_inventory_constraint = pyo.Constraint(
            self.T, rule=rule, doc="Maximum inventory constraint"
        )

    def set_sufficient_inventory_constraint(self):
        def rule(model, t):
            return (
                sum(model.delta[i-1] * model.p[i] for i in range(1, t+1))
                >= sum(model.xi[i] for i in range(t)) * model.beta
            )

        self.sufficient_inventory_constraint = pyo.Constraint(
            self.T, rule=rule, doc="Sufficient inventory constraint"
        )

    def set_objective(self):
        def rule(model):
            return sum(model.inventory(t) for t in model.T) * model.C * model.gamma

        self.objective = pyo.Objective(rule=rule, sense=pyo.minimize, doc="Total cost")

    def compute_xi(self):
        data_xi = self.data.groupby(["MONTH"]).CANTIDADCOMPRA.sum().reset_index()
        data_xi = data_xi.set_index("MONTH")
        return [data_xi.loc[i, "CANTIDADCOMPRA"] if i in data_xi.index else 0 for i in range(1,13)]

    def compute_consumption_rate(self):
        months = self.data["MONTH"].unique().tolist() + [13]
        months.sort()
        months_sep = [months[i] - months[i - 1] for i in range(1, len(months))]
        velocity = [xi_i/deltaX for xi_i, deltaX in zip(self.xi, months_sep) for _ in range(deltaX)]
        return velocity

    @staticmethod
    def process_data(data):
        data = data.copy()
        data["FECHAPEDIDO"] = pd.to_datetime(data["FECHAPEDIDO"], dayfirst=True)

        # Select useful columns
        data["YEAR"] = data["FECHAPEDIDO"].dt.year
        data["MONTH"] = data["FECHAPEDIDO"].dt.month
        data = data[["YEAR", "MONTH", "CANTIDADCOMPRA"]]

        return data.loc[data["YEAR"] == 2023]
