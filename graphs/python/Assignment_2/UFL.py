# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
import numpy as np


class UFL_Problem:
    """
    Class that represent a problem instance of the Uncapcitated Facility Location Problem
        
    Attributes
    ----------
    f : numpy array
        the yearly fixed operational costs of all facilities
    c : numpy 2-D array (matrix)
        the yearly transportation cost of delivering all demand from markets to facilities
    n_markets : int
        number of all markets.
    n_facilities : int
        number of all available locations.
    """

    def __init__(self, f, c, n_markets, n_facilities):

        self.f = f
        self.c = c
        self.n_markets = n_markets
        self.n_facilities = n_facilities

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
                        for i in range(len(asList)-1):
                            c_ij[n_row-1, n_column] = demand_i * \
                                float(asList[i])
                            n_column += 1
            n_line += 1
        return UFL_Problem(f_j, c_ij, n_markets, n_facilities)    

class UFL_Solution: 
    """
    Class that represent a solution to the Uncapcitated Facility Location Problem
        
    Attributes
    ----------
    y : numpy array
        binary array indicating whether facilities are open
    x : numpy 2-D array (matrix)
        fraction of demand from markets sourced from facilities
    instance: UFL_Problem
        the problem instance
    """ 
    
    def __init__(self, y, x, instance):
        self.y = y
        self.x = x
        self.instance = instance

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
    
class LagrangianHeuristic: 
    """
    Class used for the Lagrangian Heuristic
        
    Attributes
    ----------
    instance : UFL_Problem
        the problem instance
    """
    
    def __init__(self,instance):
        self.instance = instance
        
    def computeTheta(self,labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns 
        the optimal value of the Lagrangian problem
        """
        return
    
    def computeLagrangianSolution(self,labda):
        """
        Method that, given an array of Lagrangian multipliers computes and returns 
        the Lagrangian solution (as a UFL_Solution)
        """
        return
    
    def convertToFeasibleSolution(self,lagr_solution):
        """
        Method that, given the Lagrangian Solution computes and returns 
        a feasible solution (as a UFL_Solution)
        """
        return
        
    def updateMultipliers(self,labda_old,lagr_solution):
        """
        Method that, given the previous Lagrangian multipliers and Lagrangian Solution 
        updates and returns a new array of Lagrangian multipliers
        """
        return
    
    def runHeuristic(self):
        """
        Method that performs the Lagrangian Heuristic. 
        """
        