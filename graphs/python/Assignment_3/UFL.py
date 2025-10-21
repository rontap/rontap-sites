# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
from typing import Any

import numpy as np
from kernprof import no_op

gALPHA = .8  # change Using 0.5 as a robust alpha value


class UFL_Problem:

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
        # @claude Now calls the LagrangianHeuristic as required by assignment
        heuristic = LagrangianHeuristic(self)
        heuristic.runHeuristic()


class UFL_Solution:
    """
    Class that represent a solution to the Uncapcitated Facility Location Problem
    """

    def __init__(self, shouldOpenFac, sourcedFrac, instance):
        self.shouldOpenFac = shouldOpenFac
        self.sourcedFrac = sourcedFrac
        self.inst = instance

    def isFeasible(self):
        """
        Method that checks whether the solution is feasible
        """
        # @claude Check if each customer is assigned to exactly one facility
        for i in range(self.inst.n_markets):
            total_assignment = np.sum(self.sourcedFrac[i, :])
            # @claude Allow small numerical tolerance
            if abs(total_assignment - 1.0) > 1e-6:
                return False

        # @claude Check if customers are only assigned to open facilities
        for i in range(self.inst.n_markets):
            for j in range(self.inst.n_facilities):
                if self.sourcedFrac[i, j] > 1e-6 and not self.shouldOpenFac[j]:
                    return False

        return True

    def getCosts(self):
        """
        Method that computes and returns the costs of the solution
        """
        # @claude Calculate fixed costs for open facilities
        fixed_costs = np.sum(self.inst.fixed * self.shouldOpenFac)

        # @claude Calculate assignment costs
        assignment_costs = np.sum(self.inst.cost * self.sourcedFrac)

        return fixed_costs + assignment_costs


class LagrangianHeuristic:
    def __init__(self, instance):
        self.instance = instance

    def computeTheta(self, labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns
        the optimal value of the Lagrangian problem
        """
        # @claude This is the lower bound calculation from getLower()
        # @claude We need to compute it for a given lambda and Lagrangian solution
        # @claude For now, this needs the Lagrangian solution to be computed first
        # @claude So we compute the solution internally
        lagr_solution = self.computeLagrangianSolution(labda)

        term1 = np.sum(labda)  # sum over customers
        term2 = np.sum(self.instance.fixed * lagr_solution.shouldOpenFac)  # change sum(f_j * y_j)

        term3_cost = np.sum(
            self.instance.cost * lagr_solution.sourcedFrac)  # @comment Sum of assignment costs: sum(c_ij * x_ij)

        term4_lambda_subtracted = np.sum(
            labda * np.sum(lagr_solution.sourcedFrac, axis=1))  # @comment Term to subtract: sum(lambda_i * sum_j(x_ij))

        # LB = sum(lambda_i) + sum(f_j * y_j) + sum(c_ij * x_ij) - sum(lambda_i * sum_j(x_ij))
        return term1 + term2 + term3_cost - term4_lambda_subtracted  # @comment Corrected LB calculation to prevent negative values

    def computeLagrangianSolution(self, labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns
        the Lagrangian solution (as a UFL_Solution)
        """
        # @claude Initialize solution arrays
        shouldOpenFac = np.zeros(self.instance.n_facilities)
        sourcedFrac = np.zeros((self.instance.n_markets, self.instance.n_facilities))

        # @claude This is the subproblem solving logic from solve()
        for i in range(self.instance.n_facilities):
            # Reduced cost for assigning customers to facility i
            reduced_costs = self.instance.cost[:, i] - labda

            # Customers whose reduced cost is negative
            neg_customers = reduced_costs < 0

            # Compute total gain if facility i were open
            total_gain = np.sum(reduced_costs[neg_customers])

            # If opening yields negative total cost (worth it), open it
            if self.instance.fixed[i] + total_gain < 0:
                shouldOpenFac[i] = True
                sourcedFrac[
                    neg_customers, i] = 1.0  # change Correct assignment indexing (already set up this way, but confirming)

        return UFL_Solution(shouldOpenFac, sourcedFrac, self.instance)

    def convertToFeasibleSolution(self, lagr_solution):
        """
        Method that, given the Lagrangian Solution computes and returns
        a feasible solution (as a UFL_Solution)
        """
        # @claude This is the repair logic from getUpper()
        sourcedFrac = np.copy(lagr_solution.sourcedFrac)
        shouldOpenFac = np.copy(lagr_solution.shouldOpenFac)
        n_fac, n_cust = self.instance.n_facilities, self.instance.n_markets

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
                        self.instance.fixed + self.instance.cost[
                            j, :])  # change Correct cost indexing for best facility
                    shouldOpenFac[i_best] = True
                    open_facilities = np.where(shouldOpenFac)[0]
                # find the cheapest open facility for customer j
                # cost[customer j, facility set]
                i_best = open_facilities[
                    np.argmin(self.instance.cost[j, open_facilities])]  # change Correct cost indexing in UB heuristic

                sourcedFrac[j, :] = 0  # change Zero out row j
                sourcedFrac[j, i_best] = 1.0  # change Correct assignment: customer j, facility i_best

        return UFL_Solution(shouldOpenFac, sourcedFrac, self.instance)

    def updateMultipliers(self, labda_old, lagr_solution):
        """
        Method that, given the previous Lagrangian multipliers and Lagrangian Solution
        updates and returns a new array of Lagrangian multipliers
        """
        # @claude This is the lambda update logic from solve()
        # @claude We need access to best_UB which is tracked in runHeuristic
        # @claude So we pass it as an attribute

        # Calculate stable gap using the BEST UB for the step size (UB* - LB)
        stable_gap = self.best_UB - self.current_LB  # change Calculate stable gap using best_UB

        # subgradients
        # The subgradient is indexed by the relaxed constraint (market i)
        subgradient = 1.0 - np.sum(lagr_solution.sourcedFrac, axis=1)  # change Correct subgradient axis
        norm_g2 = np.dot(subgradient, subgradient)

        step_size = 0.0  # change Initialize step_size

        # --- Critical Step Size Logic ---

        # 1. Handle Near-Zero Subgradient (Numerical Stability/Convergence)
        if norm_g2 > np.finfo(float).eps:  # change Use epsilon check for robust division
            # 2. Handle First Iteration (where best_UB is inf)
            if self.is_first_iteration:  # change If it's the first run, the step size must be zero to prevent inf
                step_size = 0.0
            else:
                # 3. Handle Normal Iteration
                step_size = gALPHA * stable_gap / norm_g2
        else:
            # Subgradient is zero -> Convergence
            step_size = 0.0
            self.should_terminate = True  # change Set flag to terminate if subgradient is near zero

        # 4. Handle UB* Surpassed (Standard Subgradient Rule)
        if stable_gap < 0.0:  # change Use stable_gap to check for surpassing best known UB
            step_size = 0.0  # change Set step_size to zero if LB surpasses UB*

        # newlambdas
        assignment_sums = np.sum(lagr_solution.sourcedFrac, axis=1)  # For each customer i: ∑ⱼ xᵢⱼ
        labda_new = np.copy(labda_old)

        for i in range(self.instance.n_markets):
            if assignment_sums[i] < 1.0:
                # Under-assigned: increase λᵢ
                labda_new[i] = max(0, labda_old[i] + step_size * subgradient[i])
            elif assignment_sums[i] > 1.0:
                # Over-assigned: decrease λᵢ
                labda_new[i] = max(0, labda_old[i] + step_size * subgradient[i])
            # else: assignment_sums[i] == 1.0, keep λᵢ the same

        return labda_new

    def runHeuristic(self):
        """
        Method that performs the Lagrangian Heuristic.
        """
        global gALPHA

        # @claude Initialize multipliers
        lambdas = np.zeros(self.instance.n_markets)
        best_UB = np.inf  # change Initialize best upper bound

        # @claude These are used by updateMultipliers
        self.best_UB = best_UB
        self.current_LB = -np.inf
        self.is_first_iteration = True
        self.should_terminate = False

        for i in range(250):  # change Loop for 250 iterations
            if i % 25 == 0 and i > 0:
                gALPHA = gALPHA / 1.8

            # @claude Compute Lagrangian solution (this also computes theta internally)
            lagr_solution = self.computeLagrangianSolution(lambdas)

            # @claude Compute lower bound
            LB = self.computeTheta(lambdas)
            self.current_LB = LB

            # @claude Convert to feasible solution
            feas_solution = self.convertToFeasibleSolution(lagr_solution)

            # @claude Compute upper bound
            UB = feas_solution.getCosts()

            # Update best UB found so far
            if UB < best_UB:  # change Track and update the best UB
                best_UB = UB  # change Store the best UB
                self.best_UB = best_UB

            # Print statement must use the best_UB and re-calculate gap
            current_LB = LB  # change Get current LB
            current_Gap = best_UB - current_LB  # change Calculate gap against best UB

            print(
                f"{i} Lagrangian Solution: LB = ,{current_LB}, UBbest = ,{best_UB}, UB = ,{UB}, Gap = {current_Gap}")  # change Use tracked best UB for printout

            # @claude Update multipliers
            lambdas = self.updateMultipliers(lambdas, lagr_solution)

            # @claude After first iteration, set flag to False
            self.is_first_iteration = False

            if self.should_terminate:  # change Terminate loop if subgradient is near zero
                print("Fin")
                return

        print("Fin")