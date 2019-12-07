# ===========================================================
# GENETIC ALGORITHM IMPLEMENTATION
# PROBLEM: Travel Salesman
# Author: Juan Camilo Díaz - Data Cat - 2019
# For more detailed explanation visit: https://medium.com/datacat
# ===========================================================

import pandas as pd
import numpy as np
import random as rnd

decision_variables = [0,1,2,3,4]
distance = np.array([[0.00, 28.02,  17.12, 27.46, 46.07],
            [28.02, 0.00,  34.00, 25.55, 25.55],
            [17.12, 34.00,  0.00, 18.03, 57.38],
            [27.46, 25.55, 18.03,  0.00, 51.11],
            [46.07, 25.55, 57.38, 51.11,  0.00]])

def fitness_function(solution):
    tot_distance = 0
    for i in range(len(solution)-1):
        tot_distance += distance[decision_variables.index(solution[i]), decision_variables.index(solution[i+1])]
    return tot_distance

def initialize():
    pop_bag = []
    for i in range(10):
        rnd_sol = decision_variables.copy()
        rnd.shuffle(rnd_sol)
        pop_bag.append(rnd_sol)
    return np.array(pop_bag)

def pickOne(pop_bag):
    fit_bag_evals = eval_fit_population(pop_bag)
    a=True
    while a:
        rnIndex = rnd.randint(0, len(pop_bag)-1)
        rnPick  = fit_bag_evals["fit_wgh"][rnIndex]
        r = rnd.random()
        if  r <= rnPick:
            pickedSol = fit_bag_evals["solution"][rnIndex]
            a = False
    return pickedSol

def eval_fit_population(pop_bag):
    result = {}
    fit_vals_lst = []
    solutions = []
    for solution in pop_bag:
        fit_vals_lst.append(fitness_function(solution))
        solutions.append(solution)
    result["fit_vals"] = fit_vals_lst
    min_wgh = [np.max(list(result["fit_vals"]))-i for i in list(result["fit_vals"])]
    result["fit_wgh"]  = [i/sum(min_wgh) for i in min_wgh]
    result["solution"] = np.array(solutions)
    return result

def crossover(solA, solB):
    n = len(solA)
    child = [np.nan for i in range(n)]
    num_els = np.ceil(n*(rnd.randint(10,90)/100))
    str_pnt = rnd.randint(0, n-2)
    end_pnt = n if int(str_pnt+num_els) > n else int(str_pnt+num_els)
    blockA = list(solA[str_pnt:end_pnt])
    child[str_pnt:end_pnt] = blockA
    for i in range(n):
        if list(blockA).count(solB[i]) == 0:
            for j in range(n):
                if np.isnan(child[j]):
                    child[j] = solB[i]
                    break
    return child

def mutation(sol):
    n = len(sol)
    pos_1 = rnd.randint(0,n-1)
    pos_2 = rnd.randint(0,n-1)
    result = swap(sol, pos_1, pos_2)
    return result

def swap(sol, posA, posB):
    result = sol.copy()
    elA = sol[posA]
    elB = sol[posB]
    result[posA] = elB
    result[posB] = elA
    return result

# ========================================================
# ============ START THE EVOLUTIONARY PROCESS ============
# ========================================================

# Create the initial population bag
pop_bag  = initialize()

# Iterate over all generations
for g in range(200):

    # Calculate the fitness of elements in population bag
    pop_bag_fit = eval_fit_population(pop_bag)

    # Best individual in the current population bag
    best_fit = np.min(pop_bag_fit["fit_vals"])
    best_fit_index = pop_bag_fit["fit_vals"].index(best_fit)
    best_solution  = pop_bag_fit["solution"][best_fit_index]

    # Check if we have a new best
    if g == 0:
        best_fit_global      = best_fit
        best_solution_global = best_solution
    else:
        if best_fit <= best_fit_global:
            best_fit_global      = best_fit
            best_solution_global = best_solution

    # Create the new population bag
    new_pop_bag = []

    for i in range(10):
        # Pick 2 parents from the bag
        pA = pickOne(pop_bag)
        pB = pickOne(pop_bag)
        new_element = pA
        # Crossover the parents
        if rnd.random() <= 0.87:
            new_element = crossover(pA, pB)
        # Mutate the child
        if rnd.random() <= 0.7:
            new_element = mutation(new_element) 
        # Append the child to the bag
        new_pop_bag.append(new_element)

    # Set the new bag as the population bag
    pop_bag = np.array(new_pop_bag)


# Best fitness and solution
print(f"Best Fitness: {best_fit_global}")
print(f"Best Solution: {best_solution_global}")
