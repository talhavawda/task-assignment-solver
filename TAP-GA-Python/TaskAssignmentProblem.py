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
						into this file instead of importing the search and utils files
						-	As when I imported them and this script is executed from the command line without
							the other 2 files present in the same directory, it will not execute
						-	The Assignment Specification has given permission for the search.py file (which also imports utils.py)
							so I assumed that structuring my code in this way is allowed
							-	I have also acknowledged in my code any classes or functions that were taken from
								search.py or utils.py

"""
import copy
import random
import bisect
from datetime import datetime

"""
	Helper Functions (from utils.py)
"""

seedVal = 0

def weighted_sampler(seq, weights):
	"""
		Return a random-sample function that picks from seq weighted by (the corresponding) weights.
	"""

	totals = []

	#random.seed(datetime.now())

	global seedVal
	random.seed(seedVal)
	seedVal += 42345876

	for w in weights:
		totals.append(w + totals[-1] if totals else w)

	return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


"""
	Genetic Search Algorithm (the functions are taken from search.py)
	
	All comments are mine
	I have modified all of the functions so that the Constraints of the Task Assignment Problem are met
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



# numGenerations - number of generations of the population
def genetic_algorithm(population, fitnessFunc, genePool, f_thres=None, numGenerations=1000, crossoverRate=0.8, mutationRate=0.1):
	"""

		:param population:		The initial population of chromosomes
		:param fitnessFunc:		The fitness function used to evaluate the performance/fitness of a chromosome
		:param genePool:		The list of all possible values (all possible genes) that can be used to make an individual chromosome
		:param f_thres:
		:param numGenerations:	Number of generations of the population to generate before terminating
		:param crossoverRate:	The probability of doing cross-over (Recombination) [Should be in range 0.8-0.95]
		:param mutationRate:	The probability of doing Mutation [Should be either 0.1 or 0.001]
		:return:				the optimal Task Assignment chromosome after the fittest is found or numGenerations of populations have been generated
								and the final population that the optimal Task Assignment chromosome was taken from
	"""


	for generation in range(numGenerations):

		"""
			This is a Generational Population Model
				-	For each population generation, we generate ‘n’ children, where n is the population size, 
					and the entire population is replaced by the new one at the end of the iteration
					-	Thus, the population size remains the same throughout all the generations
		"""

		newPopulation = [] # The new generation of chromosomes

		populationsize = len(population)

		"""
			We dont always want to do Recombination and Mutation (for every individual to be placed in the new population generation)
			
			We want to do Recombination (combining 2 parents to form a child) most of the time; but sometimes we 
			want to just select an individual from the current generation and place it directly in the
			new population without mutating it or combining it with another individual
			(We dont want to add the 2 parents selected as we only want to add 1 child to the new population on each iteration)
				-	So we get a random value and if it's in the range [0, crossoverRate) then we do Recombination to
					get a child chromosome to add to the new population, otherwise the 'child' chromosome is just an
					 individual from the current population
			
			We want to do Mutation to the 'child' generated above, but very rarely
				-	We get a random value and if it's in the range [0, mutation) then we Mutate the child (before 
					adding it to the new population), otherwise we dont
		"""


		"""
			Original Code:	
					population = [mutate(recombine(*select(2, population, fitnessFunc)), genePool, mutationRate)
					  for i in range(len(population))]
		"""

		for i in range(populationsize): #Generate child i


			randNum = random.random() # A random floating value in the range [0.0, 1.0)

			child = [] # Declare the child chromosome; either generated from Recombination or taken directly from current population

			if randNum < crossoverRate: # Do Recombination (Crossover)
				parents = select(2, population, fitnessFunc) # Select 2 parents
				child = recombine(parents[0], parents[1])
			else: # Select a random individual from the current generation and place it directly in the new population
				individual = select(1, population, fitnessFunc) # Select a random individual from the current generation
				child = individual[0] # select() returns a list so get the first element from it

			randNum = random.random()  # A random floating value in the range [0.0, 1.0)

			if randNum <= mutationRate: #Do Mutation
				child = mutate(child, genePool)

			newPopulation.append(child) # Add the child to the new generation


		population = newPopulation # New population becomes the current population for the next population generation

		fittest_individual = fitness_threshold(fitnessFunc, f_thres, population)
		if fittest_individual:
			return fittest_individual, population # If the fittest possible (most optimal) chromosome is generated already (before we have done numGenerations generations) then return it


	return max(population, key=fitnessFunc), population # Return the fittest chromosome in the final population generation, and the final population


def fitness_threshold(fitness_fn, f_thres, population):
	if not f_thres:
		return None

	fittest_individual = max(population, key=fitness_fn)
	if fitness_fn(fittest_individual) >= f_thres:
		return fittest_individual

	return None


def init_population(popSize, genePool, stateLength):
	"""
		Initialises the population for the Genetic Algorithm

			This function has been modified to create the Initial Population for the Task Assignment Problem

			Create the initial population - popSize number of chromosomes
			The chromosomes are randomly generated
				i.e. each chromosome (Person-Task Assignment) in the initial population is a random assignment of Persons to Tasks
				We using Random Initialisation as it has been experimentally observed that it (random solutions) drives the population to optimality

			Each chromosome that is randomly generated meets the constraints of the problem
				- A Person can be assigned to only 1 Task
					i.e. if they are already assigned to a Task, then they cannot also be assigned to another Task,
					even if they are the best Person to perform that Task

			(See solveTaskAssignment() documentation for a description of what a chromosome is)


		:param popSize: 		The number of individuals in the population
		:param genePool: 		The list of all possible values (all possible genes) that can be used to make an individual chromosome
		:param stateLength:		The length/size of each individual chromosome (i.e. the number of Tasks for the Task Assignment Problem)
		:return: 				The initial population for this Problem to be used by the Genetic Algorithm
	"""

	g = len(genePool) # Used to select a random gene

	population = [] # list of individual chromosomes -> size will be popSize after we add all the chromosomes


	for i in range(popSize): # Create chromosome i

		#new_individual = [genePool[random.randrange(0, g)] for j in range(stateLength)]

		newIndividual = [] # Chromosome i

		"""
			A List of boolean values that says whether Person i has already been assigned to a Task or not for this chromosome
			Intialise to False, as initially no Person is assigned to a Task yet
			
			For the Task Assignment Problem, the Gene Pool is the ordered indexes of the Persons.
			So For index i: genePool[i] == i -> a value in range(g) is the index of a Person 
		"""
		isAlreadyAssigned = [False for person in range(g)]

		for j in range(stateLength): #Assign a random person (that has not been previously assigned) to perform Task j

			while True: # Repeat until we get a random Person that has not already been assigned to a Task
				randomPerson = genePool[random.randrange(0, g)] # get the index of a random Person (get a random gene)
				if isAlreadyAssigned[randomPerson] == False: # if this random person has not already been assigned to a Task
					personAssigned = randomPerson # assign this Person to perform Task j
					newIndividual.append(personAssigned)
					isAlreadyAssigned[randomPerson] = True
					break # Exit the while loop, (and go to the next Task -> start again at the beginning of this for loop block)

		population.append(newIndividual)

	return population


def select(r, population, fitness_fn):
	"""
		Select and return r parents from the population

		Selection Strategy - Roulette Selection (Fitness Proportionate Selection)
			- 	The parents are selected at random with the chance/probability of a chromosome being selected is proportional to its fitness
				-	Fitter individuals have a higher chance of being selected (for 'mating' and passing on their genes to child chromosomes)
				-	This ensures a reasonable spread of both good and bad chromosomes (with good/fit chromosomes being favoured)
			- 	This Selection Strategy was used as we want a mixture of both good and bad chromosomes
				-	If we only pick the fittest parents from the current population (Elitism Selection) then we
					lose chromosomes that are unfit but have part of the solution in them.
					It also leads to a loss of diversity and premature convergence (which is undesirable for Genetic Algorithms)
						-	Loss of Diversity is wher	e the population consists of chromosomes of similar genes,
							which can lead to having a local optimal solution, but not being able to reach the global optimal solution



		:param r: 			The number of parent chromosomes to select from the population
		:param population: 	The population of chromosomes
		:param fitness_fn: 	The fitness function used to evaluate the performance/fitness of a chromosome
		:return:			r parents (selected at fitness-proportionate random)
	"""

	"""
		A map of the chromosomes in the population where each chromosome is mapped with its corresponding fitness value (performance score)
		
		The map() function returns a map object (which is an iterator) of the results after applying the given 
		function to each item of a given iterable (list, tuple etc.)
	"""
	#fitnesses = map(fitness_fn, population)

	"""
		I changed fitnesses to be a list of fitness values where a value at an index is the fitness value of a chromosome
		at the same index in the population
			-> fitnesses[i] is the fitness value of the chromosome population[i]
	"""
	fitnesses = [fitness_fn(chromosome) for chromosome in population]

	"""
		weighted_sampler() does the Roulette Selection
		It returns a function that picks a random sample that picks from population weighted by the fitnesses
	"""
	sampler = weighted_sampler(population, fitnesses)

	# return r random samples
	return [sampler() for i in range(r)]


def recombine(x, y):
	"""
		Using Davis’ Order Crossover (OX1)  as the Recombination operator

		:param x:
		:param y:
		:return:
	"""

	"""
			Original Function did One Point Crossover:
				n = len(x)
				c = random.randrange(0, n)
				return x[:c] + y[c:]
		"""


	numTasks = len(x)
	task1 :int = -1
	task2 :int = -1

	while task1 >= task2:
		task1 = random.randrange(0, numTasks)
		task2 = random.randrange(0, numTasks)



	child = [-1 for i in range(numTasks)] # initialise child

	for task in range(task1, task2+1):
		child[task] = x[task]

	# y
	xUnused = []

	for task in range(numTasks):
		if x[task] not in child:
			xUnused.append(x[task])

	y = copy.deepcopy(y)

	n = numTasks - task2 - 1

	#copy
	yRotated = y[n:] + y[:n]

	#y1
	xUnusedOrdered = []
	for task in range(numTasks):
		if yRotated[task] in xUnused:
			xUnusedOrdered.append(yRotated[task])

	for i in range(len(xUnusedOrdered)):
		j = (task2 + i + 1) % numTasks
		child[j] = xUnusedOrdered[i]

	return  child



def recombine_uniform(x, y):
	n = len(x)
	result = [0] * n
	indexes = random.sample(range(n), n)
	for i in range(n):
		ix = indexes[i]
		result[ix] = x[ix] if i < n / 2 else y[ix]

	return ''.join(str(r) for r in result)



def mutate(chromosome, gene_pool):
	# Does Swap Mutation - 2 positions are selected in the chromosome at random, and their values are interchanged

	chromosome = copy.deepcopy(chromosome)
	gene_pool = copy.deepcopy(gene_pool)


	numTasks = len(chromosome)
	numPersons = len(gene_pool)


	#c = random.randrange(0, numTasks)
	#r = random.randrange(0, numPersons)
	#new_gene = gene_pool[r]
	#return chromosome[:c] + [new_gene] + chromosome[c + 1:]

	"""
		The mutated chromosome meets the constraint of the problem
				- A Person can be assigned to only 1 Task
					i.e. if they are already assigned to a Task, then they cannot also be assigned to another Task,
					even if they are the best Person to perform that Task

			(See solveTaskAssignment() documentation for a description of what a chromosome is)
	"""

	"""
		If the number of Persons is the same as the number of Tasks, then do Swap Mutation.
		Since we are swapping 2 genes (Persons) around in the current chromosome (Person-Task assignment),
			the problem constraint isn't violated
		
		
		However, if the number of Persons is more than the number of Tasks, then either do Swap Mutation or 
		get a random Task and swap the Person who is assigned to it with a random Person who is not assigned a Task
			-	The probability for each is 50%
		
		(Although we are told that we must assume that the number of Persons is the same as the number of Tasks,
		I am doing this validation as an extra measure.
		
		It wont be the case that there is less Persons than Tasks, as if that was the case I did validation when reading
		in the text file to remove extra Tasks, so that we have enough persons)
	"""
	if numTasks == numPersons:

		task1 = random.randrange(0, numTasks)
		task2 = random.randrange(0, numTasks)

		temp = chromosome[task1]
		chromosome[task1] = chromosome[task2]
		chromosome[task2] = temp

		return chromosome


	else:

		randNum = random.random()  # A random floating value in the range [0.0, 1.0)

		if randNum <= 0.5:  # Do Swap Mutation
			task1 = random.randrange(0, numTasks)
			task2 = random.randrange(0, numTasks)

			temp = chromosome[task1]
			chromosome[task1] = chromosome[task2]
			chromosome[task2] = temp

			return chromosome


		else: # Assign Task to an unassigned Person
			task = random.randrange(0, numTasks) # Random task to swap with a random person who is not assigned to a task

			while True:  # Repeat until we get a random Person that has not already been assigned to a Task
				randomPerson = gene_pool[random.randrange(0, numPersons)]  # get the index of a random Person (get a random gene)
				if randomPerson not in chromosome: # if randomPerson is not assigned a Task
					chromosome[task] = randomPerson
					return chromosome


# ==============================================================================
# ==============================================================================
# ==============================================================================

"""
	The Person-Task performance-values table
	Made it a global variable so that the fitness function assignmentFitness() can access it
	It gets assigned/initialised in readInputFile()
"""
PTtable = [[]]


def assignmentFitness(chromosome: list):
	"""
		The fitness/evaluation function for this Task Assignment Problem

		The fitness of a chromosome (Person-Task assignment) is the combined/total performance of the Tasks done by the Persons assigned to them
		The higher the fitness value the better the chromosome is

		This function gets called in solveTaskAssignment()

		Precondition: 	readInputFile() must already have been called in the main() function before solveTaskAssignment()
						is called, as readInputFile() initialises the PTtable variable

		:param chromosome: 	A Person-Task Assignment for this Task Assignment Problem (A list/permutation of Persons
							with the indexes being the Tasks that they are assigned to)
		:return: 			The fitness of this chromosome (the combined/total performance of the Tasks done by the Persons assigned to them)
	"""

	totalTasksPerformance = 0

	numTasks = len(chromosome)

	for task in range(numTasks):
		person = chromosome[task] # The Person assigned to this Task

		global PTtable  # PTtable in the statement variable is the global variable we defined above
		performance = PTtable[person][task]

		totalTasksPerformance += performance

	return totalTasksPerformance


def solveTaskAssignment(table: list, persons: int = None, tasks: int = None, populationSize = 100):
	"""
		Solve the Task Assignment Problem using Genetic Algorithms


		A State in the State Space for this Problem is a permutation list of integer values (where the value at
		index i in the list is the (index of the) Person (in the Person-Task table) who is assigned to perform Task i)
			- i.e. it represents an (specific) assignment of people to tasks
			- It is a permutation as a Person can only occur once in this list (see the Constraints below)

		A chromosome in a Genetic Algorithm represents a State from the State Space, thus a chromosome for
			this Task Assignment Problem is as specified above (a list of Person-to-Task assignments)
		Thus a gene is an integer value that represents the index of a Person in the Person-Task performance-values table
		The Gene Pool is the list of all possible values (all possible genes) that can be used to make an individual chromosome
			-> Thus the Gene Pool indexes of all the People (0, 1, ..., persons-1)
			-> Thus Gene Pool range: [0, persons-1]

		Constraints:
			A Person can only be assigned to 1 Task - i.e. if they are already assigned to a Task, then they cannot
				also be assigned to another Task, even if they are the best Person to perform that Task


		:param table: 	the 2D list/array of Person-Task performance-values (scores)
		:param persons: the number of Persons
		:param tasks: 	the number of Tasks
		:param populationSize	the size of the population for the Genetic Algorithms
		:return: 		the optimal Task Assignment chromosome and its associated score/fitness (as a tuple, in that order), and also the initial and final population
	"""


	if not persons:  # i.e. if persons == None (the default) - if no persons parameter was passed in
		persons = len(table)

	if not tasks:  # i.e. if tasks == None (the default) - if no tasks parameter was passed in
		tasks = len(table[0])

	"""
		The Gene Pool is the indexes of all the People:
			Their indexes are 0, 1, ..., persons-1
			
			Pseudocode:
				genePool = []
				for personIndex in [0, 1, ..., persons-1]:
					genePool.append(personIndex)
	"""
	genePool = [personIndex for personIndex in range(persons)]



	initialPopulation = init_population(populationSize, genePool, tasks)


	optimalAssignment, finalPopulation = genetic_algorithm(initialPopulation, assignmentFitness, genePool, None, 1000) # Get Fittest individual chromosome

	totalScoreOptimal = assignmentFitness(optimalAssignment)

	return optimalAssignment, totalScoreOptimal, initialPopulation, finalPopulation



# ==============================================================================


def getPersonScores(tasks: int, scoresStr: str):
	"""
		Get the list of scores (Performance Values) for a Person from a string (a line from the textfile) containing them

		:param tasks: the number of Tasks in the problem (the Person will have a score for each)
		:param scoresStr: the scores of the Person represented as a string
		:return: the scores of the Person for the Tasks represented as a list
	"""

	scores = []  # 1D list

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
		:return: the Person-Task performance-values table, number of persons, number of tasks (as a tuple, in that order)
	"""

	inputFile = open(fileName)

	persons = int(inputFile.readline())  # number of persons
	tasks = int(inputFile.readline())  # number of tasks

	"""
		UPDATE 2: Undid UPDATE
		UPDATE: if persons > tasks, we are NOT reducing the number of persons to number of tasks -> in a state/solution, some persons wont be assigned a task
		Specification: assume that the number of persons is the same as the number of tasks
		Thus if num persons != num tasks, make the number of persons be the number of tasks 
		(by reducing the one that is bigger to the one that is smaller such that we get a 'square' table; we
		can't increase the smaller parameter to the bigger one as we don't have any performance values that we can add)
	"""


	if persons > tasks:
		persons = tasks
	elif tasks > persons:
		tasks = persons

	#if tasks > persons:
	#	tasks = persons

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

	global  PTtable # PTtable in the statement variable is the global variable we defined above
	PTtable = table

	return table, persons, tasks



def printPTTable(table: list, persons: int = None, tasks: int = None):
	"""
		Display (to the terminal) the Person-Task performance-values table

		:param table: 	the 2D list/array of Person-Task performance-values (scores)
		:param persons: the number of Persons
		:param tasks: 	the number of Tasks
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


def printChromosome(chromosome):
	"""

		:param chromosome:
		:return: None
	"""

	"""
		Persons are labelled/identified using Uppercase letters of the alphabet so personIndex == 0 is  Person A (ASCII-65),
		personIndex == 1 is Person B (ASCII-66), ... and so on

		So Person of index i has ASCII value of i+65
	"""

	chromoString = "["

	for person in chromosome:  # for the person assigned to each task
		chromoString += chr(person+65)

		if person != chromosome[len(chromosome)-1]: # if not at the last element
			chromoString += ", "

	chromoString += "]"

	fitnessString = "\tFitness = " + str(assignmentFitness(chromosome))

	print(chromoString + fitnessString)


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

	print()
	print("Persons:\t", persons)
	print("Tasks:\t\t", tasks)
	print()
	print("Person-Task Performance Table:\n")
	printPTTable(table, persons, tasks)

	print("\n\n")

	populationSize = 100

	print("Population Size: ", populationSize)
	print("Number of generations: 1000")
	print()

	print("Solving the Task Assignment Problem using a Genetic Algorithm...\n")

	optimalAssignment, totalScoreOptimal, initialPopulation, finalPopulation = solveTaskAssignment(table, persons, tasks, populationSize)



	print("Intial Population: ")
	for chromosome in initialPopulation:
		printChromosome(chromosome)

	print("\n")

	print("Final Population: ")
	for chromosome in finalPopulation:
		printChromosome(chromosome)

	print("\n\n")

	print("The optimal Person-Task Assignment is: ")
	for task in range(len(optimalAssignment)):
		print("\tTask ", task+1, ":\tPerson ", chr(optimalAssignment[task]+65))

	print("\nThe total performance score for this Person-Task Assignment is: ", totalScoreOptimal)


main()
