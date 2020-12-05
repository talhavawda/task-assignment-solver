"""
			COMP301 (ARTIFICIAL INTELLIGENCE) 2020
						ASSIGNMENT 2

					Task Assignment Problem
				(Solved using a Genetic Algorithm)

				Developed by: Talha Vawda (218023210)

				This project has been developed using:
					Python 3.8.1
					PyCharm 2019.3.3 (Professional Edition) Build #PY-193.6494.30

				Acknowledgements:
					search.py
					utils.py

				Notes:
					1.	Due to the assignment's requirement of this single script being able to be run from the command line,
						I have copied the relevant code from the search.py file (and the utils.py from the AIMA GitHub repo)
						into this file instead of importing search and utils
						-	As when I imported them and this script is executed from the command line without
							the other 2 files present in the same directory, it will not execute
						-	The Assignment Specification has given permission for the search.py file (which also imports utils.py)
							so I assumed that structuring my code in this way is allowed
							-	I have also acknowledged in my code any classes or functions that were taken from
								search.py or utils.py

"""
import random
import bisect


"""
	Helper Functions (from utils.py)
"""
def weighted_sampler(seq, weights):
	"""Return a random-sample function that picks from seq weighted by weights."""
	totals = []
	for w in weights:
		totals.append(w + totals[-1] if totals else w)
	return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]



"""
	Genetic Search Algorithm (taken from search.py)
"""



def genetic_search(problem, ngen=1000, pmut=0.1, n=20):
	"""Call genetic_algorithm on the appropriate parts of a problem.
	This requires the problem to have states that can mate and mutate,
	plus a value method that scores states."""

	# NOTE: This is not tested and might not work.
	# TODO: Use this function to make Problems work with genetic_algorithm.

	s = problem.initial_state
	states = [problem.result(s, a) for a in problem.actions(s)]
	random.shuffle(states)
	return genetic_algorithm(states[:n], problem.value, ngen, pmut)


def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1):
	"""[Figure 4.8]"""
	for i in range(ngen):
		population = [mutate(recombine(*select(2, population, fitness_fn)), gene_pool, pmut)
					  for i in range(len(population))]

		fittest_individual = fitness_threshold(fitness_fn, f_thres, population)
		if fittest_individual:
			return fittest_individual

	return max(population, key=fitness_fn)


def fitness_threshold(fitness_fn, f_thres, population):
	if not f_thres:
		return None

	fittest_individual = max(population, key=fitness_fn)
	if fitness_fn(fittest_individual) >= f_thres:
		return fittest_individual

	return None


def init_population(pop_number, gene_pool, state_length):
	"""Initializes population for genetic algorithm
	pop_number  :  Number of individuals in population
	gene_pool   :  List of possible values for individuals
	state_length:  The length of each individual"""
	g = len(gene_pool)
	population = []
	for i in range(pop_number):
		new_individual = [gene_pool[random.randrange(0, g)] for j in range(state_length)]
		population.append(new_individual)

	return population


def select(r, population, fitness_fn):
	fitnesses = map(fitness_fn, population)
	sampler = weighted_sampler(population, fitnesses)
	return [sampler() for i in range(r)]


def recombine(x, y):
	n = len(x)
	c = random.randrange(0, n)
	return x[:c] + y[c:]


def recombine_uniform(x, y):
	n = len(x)
	result = [0] * n
	indexes = random.sample(range(n), n)
	for i in range(n):
		ix = indexes[i]
		result[ix] = x[ix] if i < n / 2 else y[ix]

	return ''.join(str(r) for r in result)


def mutate(x, gene_pool, pmut):
	if random.uniform(0, 1) >= pmut:
		return x

	n = len(x)
	g = len(gene_pool)
	c = random.randrange(0, n)
	r = random.randrange(0, g)

	new_gene = gene_pool[r]
	return x[:c] + [new_gene] + x[c + 1:]


# ==============================================================================
# ==============================================================================
# ==============================================================================

class TaskAssignment:
	#Take in
	def __init__(self, tablePT):
		#


def solveTaskAssignment():
	print("\nSolving the Task Assignment Problem: ")

def main():
	#a


main()

