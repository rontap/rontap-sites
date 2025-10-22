# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""

import numpy as np
import matplotlib.pyplot as plt

gALPHA = .6  # change Using 0.5 as a robust alpha value
EPSILON = 0.000001
TERMINATION_CRITERION = 100.0  # change Define a termination criterion for convergence
DAMPENING = 0.8

class Log:
    def __init__(self):
        self.UB = []
        self.LB = []
        self.bestUB = []

    def add(self, LB, UB, bestUB):
        self.UB.append(UB)
        self.LB.append(LB)
        self.bestUB.append(bestUB)

    def __str__(self):
        return "f{str(self.UB)},{str(self.LB)},{str(self.bestUB)}"

    def plot(self):
        x = range(len(self.UB))
        plt.figure(figsize=(12, 6))
        plt.plot(x, self.bestUB, color='blue', label='best_UB', linewidth=1.5)
        plt.plot(x, self.UB, color='red', label='UB', linewidth=1.5)
        plt.plot(x, self.LB, color='orange', linewidth=0.8)
        plt.xlim(0, len(self.UB))
        plt.ylim(0, self.bestUB[0] * 2)
        plt.legend(loc='upper center', ncol=3, frameon=False)
        plt.tight_layout()
        plt.show()


class UFL_Problem:

    def __init__(self, f, c, n_markets, n_facilities):
        self.fixed = f
        self.cost = c
        self.n_markets = n_markets
        self.n_facilities = n_facilities
        self.solution = None
        self.log = Log()
        self.alpha = gALPHA

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
            if abs(total_assignment - 1.0) > EPSILON:
                return False

        # @claude Check if customers are only assigned to open facilities
        for i in range(self.inst.n_markets):
            for j in range(self.inst.n_facilities):
                if self.sourcedFrac[i, j] > EPSILON and not self.shouldOpenFac[j]:
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
                sourcedFrac[neg_customers, i] = 1.0

        return UFL_Solution(shouldOpenFac, sourcedFrac, self.instance)

    def convertToFeasibleSolution(self, lagr_solution):
        """
        Simplest repair:
        - Keep opened facilities from Lagrangian solution
        - Assign EVERY customer to their nearest open facility
        - If no facilities open, open the one that minimizes total cost
        """
        shouldOpenFac = np.copy(lagr_solution.shouldOpenFac)
        sourcedFrac = np.zeros((self.instance.n_markets, self.instance.n_facilities))

        open_facilities = [j for j in range(self.instance.n_facilities) if shouldOpenFac[j]]

        # If no facilities are open, open one
        if len(open_facilities) == 0:
            # Find facility that minimizes: fixed_cost + total_assignment_cost
            total_costs = np.zeros(self.instance.n_facilities)
            for j in range(self.instance.n_facilities):
                total_costs[j] = self.instance.fixed[j] + np.sum(self.instance.cost[:, j])

            best_j = np.argmin(total_costs)
            shouldOpenFac[best_j] = True
            open_facilities = [best_j]

        # Assign every customer to nearest open facility
        for i in range(self.instance.n_markets):
            costs = self.instance.cost[i, open_facilities]
            best_idx = np.argmin(costs)
            best_facility = open_facilities[best_idx]
            sourcedFrac[i, best_facility] = 1.0

        return UFL_Solution(shouldOpenFac, sourcedFrac, self.instance)

    @staticmethod
    def openFacilities(shouldOpenFac):
        return [j for j, is_open in enumerate(shouldOpenFac) if is_open]

    def updateMultipliers(self, lambdaOg, lagr_solution):
        """
        Method that, given the previous Lagrangian multipliers and Lagrangian Solution
        updates and returns a new array of Lagrangian multipliers
        """
        # @claude This is the lambda update logic from solve()
        # @claude We need access to best_UB which is tracked in runHeuristic
        # @claude So we pass it as an attribute

        # Calculate stable gap using the BEST UB for the step size (UB* - LB)
        gap = self.best_UB - self.current_LB  # change Calculate stable gap using best_UB

        # subgradients
        # The subgradient is indexed by the relaxed constraint (market i)
        subgradient = 1.0 - np.sum(lagr_solution.sourcedFrac, axis=1)  # change Correct subgradient axis
        norm_g2 = np.dot(subgradient, subgradient)

        stepSize = 0.0  # change Initialize stepSize

        # --- Critical Step Size Logic ---

        # 1. Handle Near-Zero Subgradient (Numerical Stability/Convergence)
        if norm_g2 > EPSILON:  # change Use epsilon check for robust division
            # 2. Handle First Iteration (where best_UB is inf)
            if self.isFirstIter:  # change If it's the first run, the step size must be zero to prevent inf
                stepSize = 0.0
            else:
                # 3. Handle Normal Iteration
                stepSize = self.instance.alpha * gap / norm_g2
        else:
            # Subgradient is zero -> Convergence
            stepSize = 0.0
            self.shouldTerminate = True  # change Set flag to terminate if subgradient is near zero

        # 4. Handle UB* Surpassed (Standard Subgradient Rule)
        if gap < 0.0:  # change Use gap to check for surpassing best known UB
            stepSize = 0.0  # change Set stepSize to zero if LB surpasses UB*

        # newlambdas
        sums = np.sum(lagr_solution.sourcedFrac, axis=1)  # For each customer i: ∑ⱼ xᵢⱼ
        lambdaNext = np.copy(lambdaOg)

        for i in range(self.instance.n_markets):
            if sums[i] < 1.0:
                # Under-assigned: increase λᵢ
                lambdaNext[i] = max(0, lambdaOg[i] + stepSize * subgradient[i])
            elif sums[i] > 1.0:
                # Over-assigned: decrease λᵢ
                lambdaNext[i] = max(0, lambdaOg[i] + stepSize * subgradient[i])
            # else: sums[i] == 1.0, keep λᵢ the same

        return lambdaNext

    def runHeuristic(self):
        """
        Method that performs the Lagrangian Heuristic.
        """

        # @claude Initialize multipliers
        lambdas = np.zeros(self.instance.n_markets)
        best_UB = np.inf  # change Initialize best upper bound

        # @claude These are used by updateMultipliers
        self.best_UB = best_UB
        self.current_LB = -np.inf
        self.isFirstIter = True
        self.shouldTerminate = False

        for i in range(250):  # change Loop for 250 iterations
            if i % 25 == 0 and i > 0:
                self.instance.alpha = self.instance.alpha * DAMPENING

            # @claude Compute Lagrangian solution (this also computes theta internally)
            solutionLagrangian = self.computeLagrangianSolution(lambdas)

            # @claude Compute lower bound
            LB = self.computeTheta(lambdas)
            self.current_LB = LB

            # @claude Convert to feasible solution
            solutionFeasible = self.convertToFeasibleSolution(solutionLagrangian)

            # @claude Compute upper bound
            UB = solutionFeasible.getCosts()

            # Update best UB found so far
            if UB < best_UB:  # change Track and update the best UB
                best_UB = UB  # change Store the best UB
                self.best_UB = best_UB

            # Print statement must use the best_UB and re-calculate gap
            current_LB = LB  # change Get current LB
            currentGap = best_UB - current_LB  # change Calculate gap against best UB

            self.instance.log.add(LB, UB, best_UB)
            print(
                f"{i} Lagrangian Solution: LB = ,{current_LB:.2f}, UBbest = ,{best_UB:.2f}, UB = ,{UB:.2f}, Gap = {currentGap:.2f}")  # change Use tracked best UB for printout

            # @claude Update multipliers
            lambdas = self.updateMultipliers(lambdas, solutionLagrangian)

            # @claude After first iteration, set flag to False
            self.isFirstIter = False

            if self.shouldTerminate or abs(
                    current_LB - best_UB) < TERMINATION_CRITERION:  # change Terminate loop if subgradient is near zero
                print("Terminating early because of convergence")
                return
        print(self.instance.log.plot())
        print("Fin")
