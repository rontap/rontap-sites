# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
import numpy as np

gALPHA = 0.1  # change Using 0.5 as a robust alpha value


class UFL_Problem:
    # ... (UFL_Problem and readInstance methods are unchanged) ...

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
                    f_j = np.empty(n_facilities)  # change fixed cost array size to n_facilities
                    c_ij = np.empty((n_markets, n_facilities))
                elif n_line <= n_markets:  # For customers
                    index = n_line - 1
                    if index < n_facilities:  # To avoid index error if n_markets > n_facilities
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
        best_UB = np.inf  # change Initialize best upper bound

        # The loop must run once to establish a finite best_UB before that value is used in solve()
        # We will use a flag to handle the first iteration

        for i in range(150):  # change Loop for 12 iterations

            # --- Establish Best UB for the FIRST Iteration ONLY ---
            # If best_UB is still inf, calculate the solution *without* using best_UB for step size
            # and only calculate a step size of 0.

            # We must run the subproblem first to get a current LB and UB
            lambdas, shouldOpenFac, sourcedFrac, should_terminate, debug_info = solution.solve(best_UB,
                                                                                               i == 0)  # change Pass best_UB and is_first_iteration flag

            # Update best UB found so far
            if solution.UB < best_UB:  # change Track and update the best UB
                best_UB = solution.UB  # change Store the best UB

            # Print statement must use the best_UB and re-calculate gap
            current_LB = solution.LB  # change Get current LB
            current_Gap = best_UB - current_LB  # change Calculate gap against best UB

            # @comment Print debug info first
            #print(debug_info)

            print(
                f"{i} Lagrangian Solution: LB = {current_LB}, UB = {best_UB}, Gap = {current_Gap}")  # change Use tracked best UB for printout

            if should_terminate:  # change Terminate loop if subgradient is near zero
                print("Fin")
                return

        print("Fin")


class UFL_Solution:
    """
    Class that represent a solution to the Uncapcitated Facility Location Problem
    """

    def __init__(self, shouldOpenFac, sourcedFrac, instance, lambdas):
        self.shouldOpenFac = shouldOpenFac
        self.sourcedFrac = sourcedFrac
        self.inst = instance
        self.lambdas = lambdas
        self.LB = -np.inf  # change Initialize LB attribute
        self.UB = np.inf  # change Initialize UB attribute

    def solve(self, best_UB, is_first_iteration):  # change Accept best_UB and is_first_iteration flag
        """
        Method that initializes the solution attributes
        """
        # Reset sourcedFrac for this iteration's subproblem
        self.sourcedFrac = np.zeros((self.inst.n_markets, self.inst.n_facilities))  # change Reset sourcedFrac
        self.shouldOpenFac = np.zeros(self.inst.n_facilities)  # change Reset shouldOpenFac
        should_terminate = False  # change Initialize termination flag

        for i in range(self.inst.n_facilities):
            # Reduced cost for assigning customers to facility i
            reduced_costs = self.inst.cost[:, i] - self.lambdas

            # Customers whose reduced cost is negative
            neg_customers = reduced_costs < 0

            # Compute total gain if facility i were open
            total_gain = np.sum(reduced_costs[neg_customers])

            # If opening yields negative total cost (worth it), open it
            if self.inst.fixed[i] + total_gain < 0:
                self.shouldOpenFac[i] = True
                self.sourcedFrac[
                    neg_customers, i] = 1.0  # change Correct assignment indexing (already set up this way, but confirming)

        self.LB = self.getLower()  # change Store LB in class attribute
        self.UB, a, b = self.getUpper()  # change Store UB in class attribute

        # Calculate stable gap using the BEST UB for the step size (UB* - LB)
        stable_gap = best_UB - self.LB  # change Calculate stable gap using best_UB

        # subgradients
        # The subgradient is indexed by the relaxed constraint (market i)
        subgradient = 1.0 - np.sum(self.sourcedFrac, axis=1)  # change Correct subgradient axis
        norm_g2 = np.dot(subgradient, subgradient)

        step_size = 0.0  # change Initialize step_size

        # --- Critical Step Size Logic ---

        # 1. Handle Near-Zero Subgradient (Numerical Stability/Convergence)
        if norm_g2 > np.finfo(float).eps:  # change Use epsilon check for robust division
            # 2. Handle First Iteration (where best_UB is inf)
            if is_first_iteration:  # change If it's the first run, the step size must be zero to prevent inf
                step_size = 0.0
            else:
                # 3. Handle Normal Iteration
                step_size = gALPHA * stable_gap / norm_g2
        else:
            # Subgradient is zero -> Convergence
            step_size = 0.0
            should_terminate = True  # change Set flag to terminate if subgradient is near zero

        # 4. Handle UB* Surpassed (Standard Subgradient Rule)
        if stable_gap < 0.0:  # change Use stable_gap to check for surpassing best known UB
            step_size = 0.0  # change Set step_size to zero if LB surpasses UB*

        # newlambdas
        self.lambdas = np.maximum(0, self.lambdas + step_size * subgradient)

        # @comment Debugging print statement to see convergence metrics
        debug_info = f"|--- DEBUG: LB={self.LB:.5f}, UB*={best_UB:.5f}, Gap={stable_gap:.5f}, ||g||^2={norm_g2:.2e}, t^k={step_size:.5e}"  # change Store debug info
        debug_info = ""
        return self.lambdas, self.shouldOpenFac, self.sourcedFrac, should_terminate, debug_info  # change Return termination flag and debug info

    def isFeasible(self):
        """
        Method that checks whether the solution is feasible
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
            assigned = np.where(sourcedFrac[j, :] > 0)[0]  # change Correct sourcedFrac indexing for assignment check

            is_assigned_and_open = False  # change Initialize check flag
            if len(assigned) > 0:  # change Check if any facility assigned to customer j
                # Check if the facility assigned in the subproblem is an open facility
                if np.any(shouldOpenFac[assigned]):  # change Check if any assigned facility is open
                    is_assigned_and_open = True  # change Set flag if assigned and open

            # If not assigned/open, perform repair
            if not is_assigned_and_open:  # change Use check flag for repair condition
                # find cheapest open facility
                open_facilities = np.where(shouldOpenFac)[0]
                if len(open_facilities) == 0:
                    # open cheapest facility for this customer
                    # The cost is cost[customer j, facility i]
                    i_best = np.argmin(
                        self.inst.fixed + self.inst.cost[j, :])  # change Correct cost indexing for best facility
                    shouldOpenFac[i_best] = True
                    open_facilities = np.where(shouldOpenFac)[0]
                # find the cheapest open facility for customer j
                # cost[customer j, facility set]
                i_best = open_facilities[
                    np.argmin(self.inst.cost[j, open_facilities])]  # change Correct cost indexing in UB heuristic

                sourcedFrac[j, :] = 0  # change Zero out row j
                sourcedFrac[j, i_best] = 1.0  # change Correct assignment: customer j, facility i_best

        # Compute feasible cost (upper bound)
        UB = np.sum(self.inst.fixed * shouldOpenFac) + np.sum(self.inst.cost * sourcedFrac)
        return UB, shouldOpenFac, sourcedFrac


class LagrangianHeuristic:
    # ... (LagrangianHeuristic class unchanged) ...
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
