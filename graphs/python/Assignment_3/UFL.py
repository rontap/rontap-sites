# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
@modified by Aron Tatai
"""

import numpy as np
import matplotlib.pyplot as plt

# == CONTROL PARAMETERS ===
INIT_ALPHA = 0.8  # Initial alpha value for step size calculation
EPSILON = 0.000001  # compare small numbers
TERMINATION_CRITERION = 100.0  # Terminate when UB-LB < this value
DAMPENING = 0.7  # every N iteration multiply alpha by this
DAMPENING_ITER = 25  # dampen every N iterations
ITERS = 200  # total iterations to run


class Log:
    """
    Simple class to log the progress of the algorithm and then draw
    """

    def __init__(self):
        self.UB = []
        self.LB = []
        self.bestUB = []

    def add(self, LB, UB, bestUB):
        self.UB.append(UB)
        self.LB.append(LB)
        self.bestUB.append(bestUB)

    def __str__(self):
        return f"{str(self.UB)},{str(self.LB)},{str(self.bestUB)}"

    def plot(self):
        x = range(len(self.UB))
        plt.figure(figsize=(12, 6))
        plt.plot(x, self.UB, color='red', label='UB (Upper bound)', linewidth=1.5)
        plt.plot(x, self.bestUB, color='blue', label='best_UB (Best upper bound)', linewidth=1.2)
        plt.plot(x, self.LB, color='orange', linewidth=1.5, label='LB (Lower bound)')
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
        self.alpha = INIT_ALPHA

    def __str__(self):
        return f" Uncapacitated Facility Location Problem: {self.n_markets} markets, {self.n_facilities} facilities"

    def readInstance(fileName):
        """
        Read the instance fileName
        """
        # Read filename
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
                    f_j = np.empty(n_facilities)
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
        self.isFirstIter = None
        self.current_LB = None
        self.best_LB = None
        self.best_UB = None
        self.shouldTerminate = None
        self.instance = instance

    def computeTheta(self, labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns
        the Lagrangian lower bound theta(labda). In the task description, the min(0,XX) is handled
        inside the LagrangianSolution call
        """
        lagr_solution = self.computeLagrangianSolution(labda)
        # instead of calculating them in a loop, I calculate the terms in blocks decomposing them
        term1 = np.sum(labda)  # sum over customers
        term2 = np.sum(self.instance.fixed * lagr_solution.shouldOpenFac)  # fixed costs f_j*x_j
        term3_cost = np.sum(self.instance.cost * lagr_solution.sourcedFrac)  # sum of costs, c_ij*x_ij
        term4_lambda_subtracted = np.sum(
            labda * np.sum(lagr_solution.sourcedFrac, axis=1))  # sum of lagrangian lambda_i * sum(j:x_ij)

        return np.double(term1 + term2 + term3_cost - term4_lambda_subtracted)

    def computeLagrangianSolution(self, labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns
        the Lagrangian solution (as a UFL_Solution)
        """
        # init n and n×m arrays
        shouldOpenFac = np.zeros(self.instance.n_facilities)
        sourcedFrac = np.zeros((self.instance.n_markets, self.instance.n_facilities))

        # for each facility, decide whether to open it
        for i in range(self.instance.n_facilities):
            # how much is the cost difference?
            reduced_costs = self.instance.cost[:, i] - labda
            # filter those with reduced cost
            neg_customers = reduced_costs < 0
            # get total savings from opening facility i
            totalSavings = np.sum(reduced_costs[neg_customers])

            # open facility if the total savings overcomes the fixed cost
            if self.instance.fixed[i] + totalSavings < 0:
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

        # work with opened facilities
        openFacilities = [j for j in range(self.instance.n_facilities) if shouldOpenFac[j]]

        # If no facilities are open, open one
        if len(openFacilities) == 0:
            # Find facility that minimizes: fixed_cost + total_assignment_cost
            total_costs = np.zeros(self.instance.n_facilities)
            for j in range(self.instance.n_facilities):
                total_costs[j] = self.instance.fixed[j] + np.sum(self.instance.cost[:, j])

            best_j = np.argmin(total_costs)
            shouldOpenFac[best_j] = True
            openFacilities = [best_j]

        # Assign every customer to nearest open facility
        for i in range(self.instance.n_markets):
            costs = self.instance.cost[i, openFacilities]
            idx = np.argmin(costs)  # found using AI, returns the index of the minimum cost
            bestFacility = openFacilities[idx]
            sourcedFrac[i, bestFacility] = 1.0

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
        # Idea from assignment: "To ensure convergence, it is advised that the updates
        # to the multipliers become smaller as the algorithm progresses."
        finalStepSize = stepSize * self.instance.alpha ** 2
        for i in range(self.instance.n_markets):
            if sums[i] < 1.0:
                # Under-assigned: increase λ
                lambdaNext[i] = max(0, lambdaOg[i] + finalStepSize * subgradient[i])
            elif sums[i] > 1.0:
                # Over-assigned: decrease λ
                lambdaNext[i] = max(0, lambdaOg[i] + finalStepSize * subgradient[i])
            # else: keep λ the same

        return lambdaNext

    def runHeuristic(self):
        """
        Method that performs the Lagrangian Heuristic.
        """
        lambdas = np.zeros(self.instance.n_markets)  # initialize multipliers to 0
        self.best_UB = np.inf
        self.current_LB = -np.inf
        self.best_LB = -np.inf
        self.isFirstIter = True
        self.shouldTerminate = False  # should the algorithm terminate early (convergence)

        # === MAIN ITERATION ====
        for i in range(ITERS):
            if i % DAMPENING_ITER == 0 and i > 0:  # every N iterations, dampen the value of alpha.
                self.instance.alpha = self.instance.alpha * DAMPENING

            solutionLagrangian = self.computeLagrangianSolution(lambdas)
            solutionFeasible = self.convertToFeasibleSolution(solutionLagrangian)

            LB = self.computeTheta(lambdas)
            UB = solutionFeasible.getCosts()
            self.current_LB = LB

            if UB < self.best_UB:
                self.best_UB = UB
            if LB > self.best_LB:
                self.best_LB = LB  # we only keep this for display purposes

            currentGap = UB - LB
            bestGap = self.best_UB - self.best_LB
            lambdas = self.updateMultipliers(lambdas, solutionLagrangian)
            self.isFirstIter = False

            self.instance.log.add(self.best_LB, UB, self.best_UB)
            print(
                f"{i} Lagrangian Solution: LB = ,{self.best_LB:.2f}, UBbest = ,{self.best_UB:.2f}, UB = ,{UB:.2f}, Gap = {currentGap:.2f}")  # change Use tracked best UB for printout

            if self.shouldTerminate or bestGap < TERMINATION_CRITERION:
                print("Terminating early because of convergence")
                print(self.instance.log.plot())
                return

        # ^^ END OF MAIN ITERATION
        print(self.instance.log.plot())
        print("Terminated because out of iterations")
