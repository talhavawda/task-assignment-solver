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
					search.py 	(https://github.com/aimacode/aima-python/blob/master/search.py)
					utils.py	(https://github.com/aimacode/aima-python/blob/master/utils.py)

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

# ngen - number of generations of the population
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
	"""
		map() function returns a map object(which is an iterator) of the results after applying the given function to each item of a given iterable (list, tuple etc.)
		Params:
			fun : It is a function to which map passes each element of given iterable.
			iter : It is a iterable which is to be mapped.
	"""

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


# Todo
def solveTaskAssignment(table: list, persons: int = None, tasks: int = None):
	if not persons:  # i.e. if persons == None (the default) - if no persons parameter was passed in
		persons = len(table)

	if not tasks:  # i.e. if tasks == None (the default) - if no tasks parameter was passed in
		tasks = len(table[0])

	optimalAssignment = []

	#totalScoreOptimal = fitness_funct(optimalAssignment)

	#return optimalAssignment, totalScoreOptimal


# ==============================================================================


def getPersonScores(tasks: int, scoresStr: str):
	"""
		Get the list of scores (Performance Values) for a Person from a string (a line from the textfile) containing them

		:param tasks: the number of Tasks in the problem (the Person will have a score for each)
		:param scoresStr: the scores of the Person represented as a string
		:return: the scores of the Person for the Tasks represented as a list
	"""

	scores: int = []  # 1D list

	for i in range(tasks):  # i traverses [0, ..., tasks-1]

		scoresStr = scoresStr.strip()  # Remove leading and trailing whitespaces so that the beginning of the string is the score for this task

		"""
			The index of the first whitespace in the string
			Score values are separated by at least one whitespace, so this variable is the index after the end of the 
			value of the current task in the string, and we can use it to extract the value of the current task from the string
			(The value of the current task will be the first value in the string as we delete/remove the values 
			from the string once we finish extract them)
		"""
		whitespaceIndex = scoresStr.find(" ")

		"""
			If we are at the last task (i == tasks-1), then  there wont be a whitespace after the 
			score for this task in the scoresStr (whitespaceIndex will be -1), so dont do any slicing - the score (as a string)
			for this task is the remaining scoresStr

			Else (if not the last task, thus there is a whitespace between the score for this task and the next task)
			then to get the score (as a string) for this task, slice from the beginning of the scoresStr till the character
			before the whitespace
			
			If the actual number of tasks in the textfile is more than the tasks variable (this occurs when num of tasks
			is greater than num of persons, so we reduce the number of tasks we using), this algorithm still works - since
			there'll be a whitespace (followed by a value) in the string, the else part wll execute, extracting just the value
			for this (last) task (last task based on the number of tasks we using) instead of the remainder of the string,
			and the loop will terminate (thus we don't also extract the values of the extra tasks from the textfile) 
		"""
		if whitespaceIndex == -1:
			scoreStr = scoresStr
		else:
			scoreStr = scoresStr[:whitespaceIndex]

		score = int(scoreStr)
		scores.append(score)
		scoresStr = scoresStr[whitespaceIndex + 1:]  # Remove the score for this task from scoresStr

	return scores


def readInputFile(fileName: str):
	"""
		Reads in the input textfile containing the Person-Task performance values table

		The Assignment Specification Document specifies that the first line of the textfile is
		the number of persons (rows in the table),	the second line is the number of tasks (columns in the table),
		and the remainder lines is the table/matrix (without the row and column headings; just the raw performance values)

		The lecturer mentioned in the class meeting that we must assume that the number of persons is the same as the number of tasks

		:param fileName: the name of the input textfile
		:return: the Person-Task performance-values table, number of persons, number of tasks (as a tuple in that order)
	"""

	inputFile = open(fileName)

	persons = int(inputFile.readline())  # number of persons
	tasks = int(inputFile.readline())  # number of tasks

	"""
		Specification: assume that the number of persons is the same as the number of tasks
		Thus if num persons != num tasks, make the number of persons be the number of tasks 
		(by reducing the one that is bigger to the one that is smaller such that we get a 'square' table; we
		can't increase the smaller parameter to the bigger one as we don't have any performance values that we can add)
	"""
	if persons > tasks:
		persons = tasks
	elif tasks > persons:
		tasks = persons

	"""
		This table variable is a 2D list/array which represents the Person-Task performance-values table
		It is a list of lists with each sublist being a row in the table - each sublist represents a person.
		For sublist i, the value at index j is how well Person i performs Task j (i.e. the performance value)
	"""
	table = []

	"""
		For each person, get their row (their list of scores/values for the Tasks) from the Person-Task table in the textfile
		and add it to the table 2D list
		We skip over any empty/blank lines we come across (this is to cater for blank lines between the number of tasks line
		and the start of the actual table in the textfile)
	"""
	for person in range(persons):

		while True:  # Get the next non-empty/blank line (to skip over any blank lines between the number of tasks and the table itself)
			personScoresString = inputFile.readline()
			if personScoresString != "\n":
				break

		personScores = getPersonScores(tasks, personScoresString)
		table.append(personScores)


	inputFile.close()


	return table, persons, tasks



def printPTTable(table: list, persons: int = None, tasks: int = None):
	"""
		Display (to the terminal) the Person-Task performance-values table

		:param table: 	the 2D list/array of performance values (scores)
		:param persons: number of Persons
		:param tasks: 	number of Tasks
		:return: None
	"""

	if not persons:  # i.e. if persons == None (the default) - if no persons parameter was passed in
		persons = len(table)

	if not tasks:	# i.e. if tasks == None (the default) - if no tasks parameter was passed in
		tasks = len(table[0])

	# Column Headings (Tasks)
	print("Tasks ->", end="\t\t")
	headerStr = "=============="

	#Tasks are labelled/identified using numbers/digits so first task is Task 1 (index 0), 2nd is Task 2, ..., and so on
	for task in range(tasks):
		print(task + 1, end="\t")
		headerStr += "===="

	print("\nPersons:")
	print(headerStr)


	"""
		Persons are labelled/identified using Uppercase letters of the alphabet so first person is Person A (index 0; ASCII-65),
		2nd is Person B (index 1; ASCII-66), ... and so on
		
		So Person at index i has ASCII value of 65+i
	"""
	for person in range(persons):  # for each row
		print("\t", chr(person+65), end="\t|\t\t")
		for task in table[person]:  # for each column in a row
			print(task, end="\t") # Display the performance value of this person doing this task
		print()



def main():
	"""
		The main function of the program
		It gets the input textfile name from the user, reads in the textfile to get the Person-Task performance values table,
		solves the Task Assignment Problem, and displays the results (by calling the relevant functions)

	:return: None

	"""
	print("\n\nSolving the Task Assignment Problem using Genetic Algorithms")
	print("=============================================================\n\n")

	fileName = "input.txt"  # Set the default filename to 'input.txt'
	print("Please enter the file name of the Person-Task performance values")
	print(
		"Note:\tIf no absolute path to the file is specified, the file is assumed to be in the current working directory")
	print("\t\tIf the Enter key is pressed without entering any filename, 'input.txt' will be used as the file name")
	userFile = input("File Name: ")

	userFile = userFile.strip();

	if userFile:  # if the user-entered file name is NOT the empty string
		fileName = userFile

	table, persons, tasks = readInputFile(fileName)

	#optimalAssignment, totalScoreOptimal = solveTaskAssignment(table, persons, tasks)

	print()
	print("Persons:\t", persons)
	print("Tasks:\t\t", tasks)
	print()
	print("Person-Task Performance Table:\n")
	printPTTable(table, persons, tasks)

	print("\n")
	print("The optimal Task Assignment is: ")
	print("The total score for this assignment is: ")


main()
