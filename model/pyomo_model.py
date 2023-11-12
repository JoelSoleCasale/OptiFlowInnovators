import pyomo.environ as pyo
import pandas as pd


class PartialModel(pyo.ConcreteModel):
    """
    Partial model for the optimization problem of a given product


    Args:
        data (pd.DataFrame): data for a given product
        beta (float): percentage of demand to be satisfied
        Cmax (int): maximum inventory
        C (float): cost of inventory per unit (per month)
        delta (list): list of months to produce

    Attributes:
        data (pd.DataFrame): data for a given product
        beta (float): percentage of demand to be satisfied
        Cmax (int): maximum inventory
        C (float): cost of inventory per unit
        T (pyo.Set): time periods
        p (pyo.Var): production amount
        xi (list): demand for each month
        v (list): consumption rate for each month
        nonegative_inventory_constraint (pyo.Constraint): non-negative inventory constraint
        max_inventory_constraint (pyo.Constraint): maximum inventory constraint
        sufficient_inventory_constraint (pyo.Constraint): sufficient inventory constraint
        objective (pyo.Objective): total cost
    """

    def __init__(
        self,
        data: pd.DataFrame,
        beta: float,
        Cmax: int,
        C: float,
        delta: list = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data = self.process_data(data)
        self.beta = beta
        self.Cmax = Cmax
        self.C = C
        self.delta = delta
        self.build_model()

    def build_model(self) -> None:
        """
        Build the model.

        Returns:
            None
        """
        self.set_indices()
        self.set_parameters()
        self.set_variables()
        self.set_constraints()
        self.set_objective()

    def set_indices(self) -> None:
        """
        Set the indices of the model.

        Returns:
            None
        """
        self.T = pyo.Set(initialize=range(1, 13), doc="Time periods")

    def set_variables(self) -> None:
        """
        Set the variables of the model.

        Returns:
            None
        """
        self.p = pyo.Var(
            self.T, domain=pyo.NonNegativeReals, doc="Production amount", initialize=0
        )

    def set_parameters(self) -> None:
        """
        Computes the parameters xi and v of the model.

        Returns:
            None
        """
        self.xi = self.compute_xi()
        self.v = self.compute_consumption_rate()

    def set_constraints(self) -> None:
        """
        Set the constraints of the model.
        
        Returns:
            None"""
        self.set_inventory_constraint()

    def set_inventory_constraint(self) -> None:
        """
        Set the inventory constraints.
        
        Returns:
            None
        """
        self.set_nonegative_inventory_constraint()
        self.set_max_inventory_constraint()
        self.set_sufficient_inventory_constraint()

    def inventory(self, t) -> pyo.Var:
        """
        Compute the inventory at time t.

        Args:
            t (int): time period (month)

        Returns:
            inventory (pyo.Var): inventory at time t
        """
        return (
            sum(self.delta[i - 1] * self.p[i] - self.v[i] for i in range(1, t))
            + self.delta[t - 1] * self.p[t]
        )

    def set_nonegative_inventory_constraint(self) -> None:
        """
        Set the non-negative inventory constraint.

        Returns:
            None
        """
        def rule(model, t):
            return model.inventory(t) >= 0

        self.nonegative_inventory_constraint = pyo.Constraint(
            self.T, rule=rule, doc="Non-negative inventory constraint"
        )

    def set_max_inventory_constraint(self) -> None:
        """
        Set the maximum inventory constraint using the Cmax parameter.

        Returns:
            None
        """
        def rule(model, t):
            return model.inventory(t) <= self.Cmax

        self.max_inventory_constraint = pyo.Constraint(
            self.T, rule=rule, doc="Maximum inventory constraint"
        )

    def set_sufficient_inventory_constraint(self) -> None:
        """
        Set the sufficient inventory constraint.
        
        Returns:
            None
        """
        def rule(model, t):
            return (
                sum(model.delta[i - 1] * model.p[i] for i in range(1, t + 1))
                >= sum(model.xi[i] for i in range(t)) * model.beta
            )

        self.sufficient_inventory_constraint = pyo.Constraint(
            self.T, rule=rule, doc="Sufficient inventory constraint"
        )

    def set_objective(self) -> None:
        """
        Set the objective function of the model.

        Returns:
            None
        """

        def rule(model):
            return sum(model.inventory(t) for t in model.T) * model.C

        self.objective = pyo.Objective(rule=rule, sense=pyo.minimize, doc="Total cost")

    def compute_xi(self) -> list:
        """
        Compute the demand for each month.

        Returns:
            xi (list): demand for each month
        """
        data_xi = self.data.groupby(["MONTH"]).CANTIDADCOMPRA.sum().reset_index()
        data_xi = data_xi.set_index("MONTH")
        return [
            data_xi.loc[i, "CANTIDADCOMPRA"] if i in data_xi.index else 0
            for i in range(1, 13)
        ]

    def compute_consumption_rate(self) -> list:
        """
        Compute the consumption rate for each month.
        
        Returns:
            velocity (list): consumption rate for each month
        """
        months = self.data["MONTH"].unique().tolist() + [13]
        months.sort()
        months_sep = [months[i] - months[i - 1] for i in range(1, len(months))]
        velocity = [
            xi_i / deltaX
            for xi_i, deltaX in zip(self.xi, months_sep)
            for _ in range(deltaX)
        ]
        return velocity

    @staticmethod
    def process_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Process data to be used in the model.

        Args:
            data (pd.Dataframe): raw data for a given product

        Returns:
            cleaned data (pd.DataFrame)
        """
        data = data.copy()
        data["FECHAPEDIDO"] = pd.to_datetime(data["FECHAPEDIDO"], dayfirst=True)

        # Select useful columns
        data["YEAR"] = data["FECHAPEDIDO"].dt.year
        data["MONTH"] = data["FECHAPEDIDO"].dt.month
        data = data[["YEAR", "MONTH", "CANTIDADCOMPRA"]]

        return data.loc[data["YEAR"] == 2023]
