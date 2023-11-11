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
        self.data = data
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
        self.T = pyo.Set(initialize=range(13), doc="Time periods")

    def set_variables(self):
        self.p = pyo.Var(
            self.T, domain=pyo.NonNegativeReals, doc="Production amount", initialize=0
        )

    def set_parameters(self):
        self.xi = self.compute_xi()
        self.v = self.compute_consumption_rate()

    def set_constraints(self):
        self.set_inventory_constraint()
        # self.set_order_constraint()

    def set_inventory_constraint(self):
        self.set_nonegative_inventory_constraint()
        self.set_max_inventory_constraint()
        self.set_sufficient_inventory_constraint()

    def inventory(self, t):
        if t == 0:
            return 0
        return (
            sum(self.delta[i] * self.p[i] - self.v[i] for i in range(t - 1))
            + self.delta[t - 1] * self.p[t - 1]
        )

    def set_nonegative_inventory_constraint(self):
        def rule(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.inventory(t) >= 0

        self.nonegative_inventory_constraint = pyo.Constraint(
            self.T, rule=rule, doc="Non-negative inventory constraint"
        )

    def set_max_inventory_constraint(self):
        def rule(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.inventory(t) <= self.Cmax

        self.max_inventory_constraint = pyo.Constraint(
            self.T, rule=rule, doc="Maximum inventory constraint"
        )

    def set_sufficient_inventory_constraint(self):
        def rule(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return (
                sum(model.delta[i] * model.p[i] for i in range(t))
                >= sum(model.xi[i] for i in range(t)) * model.beta
            )

        self.sufficient_inventory_constraint = pyo.Constraint(
            self.T, rule=rule, doc="Sufficient inventory constraint"
        )

    def set_order_constraint(self):
        def rule(model):
            return sum(model.delta[i] for i in self.T) == self.Pmax

        self.order_constraint = pyo.Constraint(rule=rule, doc="Order constraint")

    def set_objective(self):
        def rule(model):
            return sum(model.inventory(t) for t in model.T) * model.C * model.gamma

        self.objective = pyo.Objective(rule=rule, sense=pyo.minimize, doc="Total cost")

    def compute_xi(self):
        self.data["FECHAPEDIDO"] = pd.to_datetime(data["FECHAPEDIDO"], dayfirst=True)

        # Select useful columns
        data = self.generate_date_features(data)
        data = data[["CODIGO", "YEAR", "MONTH", "CANTIDADCOMPRA"]]

        # Select dates 2023
        data = data.loc[data["YEAR"] == 2023]

        # Create dataframe with xi values
        data_xi = data.groupby(["CODIGO", "MONTH"]).CANTIDADCOMPRA.sum().reset_index()
        data_xi = data_xi.sort_values(["CODIGO", "MONTH"], ascending=[True, True])

        return data_xi["CANTIDADCOMPRA"].tolist()

    def compute_consumption_rate(self):
        velocity = self.xi.copy()
        velocity["VELOCITY"] = 0
        for i in range(0, len(velocity)):
            if (i + 1 < len(velocity)) and (
                velocity.iloc[i, 0] == velocity.iloc[i + 1, 0]
            ):
                velocity.iloc[i, 3] = velocity.iloc[i, 2] / (
                    velocity.iloc[i + 1, 1] - velocity.iloc[i, 1]
                )
            else:
                velocity.iloc[i, 3] = velocity.iloc[i, 2] / (13 - velocity.iloc[i, 1])
        return velocity["VELOCITY"].tolist()

    @staticmethod
    def generate_date_features(data):
        data["YEAR"] = data["FECHAPEDIDO"].dt.year
        data["MONTH"] = data["FECHAPEDIDO"].dt.month
        return data
