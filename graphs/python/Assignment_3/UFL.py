# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
import numpy as np

gALPHA = 1.5


class UFL_Problem:
    """
    Class that represent a problem instance of the Uncapcitated Facility Location Problem
        
    Attributes
    ----------
    fixed : numpy array
        the yearly fixed operational costs of all facilities
    cost : numpy 2-D array (matrix)
        the yearly transportation cost of delivering all demand from markets to facilities
    n_markets : int
        number of all markets.
    n_facilities : int
        number of all available locations.
    """

    def __init__(self, f, c, n_markets, n_facilities):

        self.fixed = f
        self.cost = c
        self.n_markets = n_markets
        self.n_facilities = n_facilities
        self.solution = None

    def __str__(self):
        return f" Uncapacitated Facility Location Problem: {self.n_markets} markets, {self.n_facilities} facilities"

    def readInstance(fileName):
        """
        Read the instance fileName

        Parameters
        ----------
        fileName : str
            instance name in the folder Instance.

        Returns
        -------
        UFL Object

        """
        # Read filename
        print("read instance")
        f = open(f"Instances/{fileName}")
        n_line = 0
        n_markets = 0
        n_facilities = 0
        n_row = 0
        for line in f.readlines():
            asList = line.replace(" ", "_").split("_")
            if line:
                if n_line == 0:
                    n_markets = int(asList[0])
                    n_facilities = int(asList[1])
                    f_j = np.empty(n_markets)
                    c_ij = np.empty((n_markets, n_facilities))
                elif n_line <= n_markets:  # For customers
                    index = n_line - 1
                    f_j[index] = asList[1]
                else:
                    if len(asList) == 1:
                        n_row += 1
                        demand_i = float(asList[0])
                        n_column = 0
                    else:
                        for i in range(len(asList) - 1):
                            c_ij[n_row - 1, n_column] = demand_i * \
                                                        float(asList[i])
                            n_column += 1
            n_line += 1
        return UFL_Problem(f_j, c_ij, n_markets, n_facilities)

    def solve(self):
        initShouldOpenFac = np.zeros(self.n_facilities)
        initSourcedFrac = np.zeros((self.n_markets, self.n_facilities))
        lambdas = np.zeros(self.n_markets)
        solution = UFL_Solution(initShouldOpenFac, initSourcedFrac, self, lambdas)
        for i in range(10):
            lambdas, shouldOpenFac, sourcedFrac = solution.solve()
        lambdas, shouldOpenFac, sourcedFrac = solution.solve()


class UFL_Solution:
    """
    Class that represent a solution to the Uncapcitated Facility Location Problem
        
    Attributes
    ----------
    shouldOpenFac : numpy array
        binary array indicating whether facilities are open
    sourcedFrac : numpy 2-D array (matrix)
        fraction of demand from markets sourced from facilities
    instance: UFL_Problem
        the problem instance
    """

    def __init__(self, shouldOpenFac, sourcedFrac, instance, lambdas):
        self.shouldOpenFac = shouldOpenFac
        self.sourcedFrac = sourcedFrac
        self.inst = instance
        self.lambdas = lambdas

    def solve(self):
        """
        Method that initializes the solution attributes
        """
        for i in range(self.inst.n_facilities):
            # Reduced cost for assigning customers to facility i
            reduced_costs = self.inst.cost[i, :] - self.lambdas

            # Customers whose reduced cost is negative
            neg_customers = reduced_costs < 0

            # Compute total gain if facility i were open
            total_gain = np.sum(reduced_costs[neg_customers])

            # If opening yields negative total cost (worth it), open it
            if self.inst.fixed[i] + total_gain < 0:
                self.shouldOpenFac[i] = True
                self.sourcedFrac[i, neg_customers] = 1.0

        self.LB = self.getLower()
        self.UB, a, b = self.getUpper()
        self.gap = self.UB - self.LB
        print(f"Initial Lagrangian Solution: LB = {self.LB}, UB = {self.UB}, Gap = {self.gap}")

        # subgradients

        subgradient = 1.0 - np.sum(self.sourcedFrac, axis=0)
        norm_g2 = np.dot(subgradient, subgradient)
        if norm_g2 > 0:
            step_size = gALPHA * self.gap / norm_g2
        else:
            step_size = 0.0
        # newlambdas
        self.lambdas = np.maximum(0, self.lambdas + step_size * subgradient)
        return self.lambdas, self.shouldOpenFac, self.sourcedFrac

    def isFeasible(self):
        """
        Method that checks whether the solution is feasible
        
        Returns true if feasible, false otherwise
        """
        return

    def getCosts(self):
        """
        Method that computes and returns the costs of the solution
        """
        return

    def getLower(self):
        """
        Method that computes and returns the Lagrangian costs of the solution
        """
        term1 = np.sum(self.lambdas)  # sum over customers
        term2 = np.sum(self.inst.fixed * self.shouldOpenFac)
        term3 = np.sum((self.inst.cost - self.lambdas) * self.sourcedFrac)
        return term1 + term2 + term3

    def getUpper(self):
        sourcedFrac = np.copy(self.sourcedFrac)
        shouldOpenFac = np.copy(self.shouldOpenFac)
        n_fac, n_cust = self.inst.n_facilities, self.inst.n_markets

        # Ensure each customer assigned to an open facility
        for j in range(n_cust):
            assigned = np.where(sourcedFrac[:, j] > 0)[0]
            if len(assigned) == 0 or not shouldOpenFac[assigned[0]]:
                # find cheapest open facility
                open_facilities = np.where(shouldOpenFac)[0]
                if len(open_facilities) == 0:
                    # open cheapest facility for this customer
                    i_best = np.argmin(self.inst.fixed + self.inst.cost[:, j])
                    shouldOpenFac[i_best] = True
                    open_facilities = [i_best]
                i_best = open_facilities[np.argmin(self.inst.cost[open_facilities, j])]
                sourcedFrac[:, j] = 0
                sourcedFrac[i_best, j] = 1.0

        # Compute feasible cost (upper bound)
        UB = np.sum(self.inst.fixed * shouldOpenFac) + np.sum(self.inst.cost * sourcedFrac)
        return UB, shouldOpenFac, sourcedFrac


class LagrangianHeuristic:
    """
    Class used for the Lagrangian Heuristic
        
    Attributes
    ----------
    instance : UFL_Problem
        the problem instance
    """

    def __init__(self, instance):
        self.instance = instance

    def computeTheta(self, labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns 
        the optimal value of the Lagrangian problem
        """
        return

    def computeLagrangianSolution(self, labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns 
        the Lagrangian solution (as a UFL_Solution)
        """
        return

    def convertToFeasibleSolution(self, lagr_solution):
        """
        Method that, given the Lagrangian Solution computes and returns 
        a feasible solution (as a UFL_Solution)
        """
        return

    def updateMultipliers(self, labda_old, lagr_solution):
        """
        Method that, given the previous Lagrangian multipliers and Lagrangian Solution 
        updates and returns a new array of Lagrangian multipliers
        """
        return

    def runHeuristic(self):
        """
        Method that performs the Lagrangian Heuristic. 
        """
