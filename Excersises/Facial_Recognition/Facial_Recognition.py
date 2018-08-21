import random
from functools import reduce


class genocode:
    def __init__(self, pile_0, pile_1):
        self.pile_0 = pile_0
        self.pile_1 = pile_1

    def __str__(self):
        return str("Pile 0: " + str(self.pile_0) + "\nPile 1: " + str(self.pile_1))

    def calculatePile(self, numberOfPile):
        #print(self.pile_0, "\t", self.pile_1)
        if numberOfPile == 1:
            return reduce(lambda x, y: x + y, self.pile_0)
        else:
            return reduce(lambda x, y: x * y, self.pile_1)

    def calculateFitness(self):
        calculationPile0 = self.calculatePile(1)
        calculationPile1 = self.calculatePile(2)
        if calculationPile0 > 36 and calculationPile1 > 360:
            return ((36 - calculationPile0)*-1 + (360 - calculationPile1)*-1)
        elif calculationPile0 > 36:
            return ((36 - calculationPile0)*-1 + (360 - calculationPile1))
        elif calculationPile1 > 360:
            return ((36 - calculationPile0) + (360 - calculationPile1)*-1)
        else:
            return ((36 - calculationPile0) + (360 - calculationPile1))
        


class genetic:
    def __init__(self):
        self.generation = []
        self.pileOfCards = [1,2,3,4,5,6,7,8,9,10]

    def generateRandomChild(self):
        childPile0 = []
        childPile1 = []
        len = random.randint(2,8)
        for i in range(0,len):
            temp = random.randint(1,10)
            while temp in childPile0 or temp in childPile1:
                temp = random.randint(1,10)
            childPile0.append(temp)

        for j in range(0, 10-len):
            temp = random.randint(1,10)
            while temp in childPile0 or temp in childPile1:
                temp = random.randint(1,10)
            childPile1.append(temp)
        return genocode(childPile0, childPile1)
        
    def generateChild(self, parent_1, parent_2):
        childPile0 = []
        childPile1 = []
        chooseParent = random.randint(0,1)
        if chooseParent == 0:
            length = len(parent_1.pile_0)
        else:
            length = len(parent_2.pile_0)
        for i in range(0,length):
            chooseParent = random.randint(0,1)
            if chooseParent == 0:
                if i > (len(parent_1.pile_0)-1):
                    temp = parent_2.pile_0[i]
                else:
                    temp = parent_1.pile_0[i]
            else:
                if i > (len(parent_2.pile_0)-1):
                    temp = parent_1.pile_0[i]
                else:
                    temp = parent_2.pile_0[i]
            while temp in childPile0 or temp in childPile1:
                temp = random.randint(1,10)
            childPile0.append(temp)

        for i in range(0,10-(length)):
            chooseParent = random.randint(0,1)
            if chooseParent == 0:
                if i > (len(parent_1.pile_1)-1):
                    temp = parent_2.pile_1[i]
                else:
                    temp = parent_1.pile_1[i]
            else:
                 if i > (len(parent_2.pile_1) -1):
                    temp = parent_1.pile_1[i]
                 else:
                    temp = parent_2.pile_1[i]
            while temp in childPile1 or temp in childPile0:
                temp = random.randint(1,10)
            childPile1.append(temp)

        mutationIndex1 = random.randint(0, len(childPile0)-1)
        mutationIndex2 = random.randint(0, len(childPile1)-1)

        temp = childPile0[mutationIndex1]
        childPile0[mutationIndex1] = childPile1[mutationIndex2]
        childPile1[mutationIndex2] = temp
        return genocode(childPile0, childPile1)



    def generateGeneration(self, sizeOfGeneration):
        newGeneration = []

        for i in range(0, sizeOfGeneration):
            indexNumbers = []
            tournements = []
            for i in range(0,4):
                temp = random.randint(0,len(self.generation) -1)
                if temp in indexNumbers:
                    temp = random.randint(0,len(self.generation) -1)
                indexNumbers.append(temp)
                tournements.append(self.generation[indexNumbers[i]].calculateFitness())
            parent1 = self.generation[tournements.index(min(tournements))]
            tournements.pop(tournements.index(min(tournements)))
            parent2 = self.generation[tournements.index(min(tournements))]
            newGeneration.append(self.generateChild(parent1, parent2))
        self.generation = newGeneration
    
    def generateFirstGeneration(self, sizeOfGeneration):
        newGeneration = []
        for i in range(0, sizeOfGeneration):
            newGeneration.append(self.generateRandomChild())
        self.generation = newGeneration

    def findBestPerson(self):
        temp = self.generation[0].calculateFitness()
        lastIndex = 0
        for i in range(0,len(self.generation)):
            if temp > self.generation[i].calculateFitness():
                temp = self.generation[i].calculateFitness()
                lastIndex = i
        return self.generation[lastIndex]


"""
6.1 Assignment:Card problem
Write a program to run a Genetic Algorithm with your genotype encoding and
fitness function. Run it once for a suitable (probably very large – you choose)
number of generations. Then repeat that same run a number of times (like, 100)
and see if you get the same answer each time, see how much variance there is
between each run.

"""
for i in range(0,0):
    t = genetic()
    t.generateFirstGeneration(50)
    for j in range(0,30):
        t.generateGeneration(50)
    print(t.findBestPerson())
    print(t.findBestPerson().calculateFitness())
    print()

"""
Answer:
    Amount of generations per run: 10
    There is a lot of variance between answers. Overall most 
    answers are between 0 and 50 but the are some outlairs (4671 for example)
    
    Amount of generations per run: 30
    Interestingly enough there is quite a devide with this one. There are almost no
    inbetween answers. It's either within 10 or over a 1000 on the fitness score. 
    This method does seem to give more correct answers then the pervious one. 
"""

class wing:
    def __init__(self, A, B, C, D):
        self.Aint = A
        self.Bint = B
        self.Cint = C
        self.Dint = D
        self.A = bin(A)[bin(A).index('b')+1:]
        temp = ""
        if(len(self.A) < 6):
            for i in range(0,6 - len(self.A)):
                temp += '0'
            self.A = temp + self.A
        self.A = list(self.A)
        self.B = bin(B)[bin(B).index('b')+1:]
        temp = ""
        if(len(self.B) < 6):
            for i in range(0,6 - len(self.B)):
                temp += '0'
            self.B = temp + self.B
        self.B = list(self.B)
        self.C = bin(C)[bin(C).index('b')+1:]
        temp = ""
        if(len(self.C) <6):
            for i in range(0,6 - len(self.C)):
                temp += '0'
            self.C = temp + self.C
        self.C = list(self.C)
        self.D = bin(D)[bin(D).index('b')+1:]
        temp = ""
        if(len(self.D) < 6):
            for i in range(0,6 - len(self.D)):
                temp += '0'
            self.D = temp + self.D
        self.D = list(self.D)

    def __str__(self):
        return str("A: " + str(self.Aint) + "\n" + "B: " + str(self.Bint) + "\n" + "C: " + str(self.Cint) + "\n" + "D: " + str(self.Dint) + "\n"  + "Total lift: " + str(self.calculateLift()) + "\n" )

    def calculateLift(self):
        return(int("".join(self.A),2) - int("".join(self.B),2))**2 + (int("".join(self.C),2) + int("".join(self.D),2))**2 - (int("".join(self.A),2)-30)**3 - (int("".join(self.C),2) - 40)**3 

    def mutation(self):
        temp = random.randint(0,5)
        if(self.A[temp] is '0'):
            self.A[temp] = '1'
        else:
            self.A[temp] = '0'
        temp = random.randint(0,5)
        if(self.B[temp] is '0'):
            self.B[temp] = '1'
        else:
            self.B[temp] = '0'
        #print(self.B)
        temp = random.randint(0,5)
        if(self.C[temp] is '0'):
            self.C[temp] = '1'
        else:
            self.C[temp] = '0'
        temp = random.randint(0,5)
        if(self.D[temp] is '0'):
            self.D[temp] = '1'
        else:
            self.D[temp] = '0'
        k = wing(int("".join(self.A), 2), int("".join(self.B), 2), int("".join(self.C), 2), int("".join(self.D),2))
        return k

class geneticAlgorithm:
    def __init__(self, number):
        self.generation = []
        for i in range(0, number):
            self.generation.append(wing(random.randint(0,63),random.randint(0,63), random.randint(0,63),random.randint(0,63)))

    def generateGeneration(self,number):
        lift = []
        generation = []
        #print(self.bestGenotype())
        for i in self.generation:
            lift.append(i.calculateLift())
        for i in range(0, number):
            generation.append(self.generation[lift.index(max(lift))].mutation())
            lift.pop(lift.index(max(lift)))
        
        self.generation = generation
        #print(self.bestGenotype())

    def bestGenotype(self):
        temp = self.generation[0]
        for i in range(0, len(self.generation)):
            if temp.calculateLift() < self.generation[i].calculateLift():
                temp = self.generation[i]
        return temp




for i in range(0,8):
    k = geneticAlgorithm(100)
    for i in range(0, 50):
        k.generateGeneration(100)
    print(k.bestGenotype())


"""
Assignment 6.2: Wing design
You have 4 variables that represent possible parameter settings for the design of an
aircraft wing. A, B, C, and D, each of which can be any whole number between 0
and 63 (use a bit encoding per parameter).
Your aerodynamics model tells you that the Lift of the wing is the following:
Lift = (A−B)^2 + (C +D)^2 −(A−30)^3 −(C −40)^3
Find values of A B C D, each within their allowed range 0-63, that maximises Lift.

Write a program to run a Genetic Algorithm with your genotype encoding and
fitness function. Run it once for a suitable (probably very large – you choose)
number of generations. Then repeat that same run a number of times (like, 100)
and see if you get the same answer each time, see how much variance there is
between each run.

Answer:

The amount of 'people' within a generation doesn't really matter. The answers seem te be 
more spread then the previous exercise. It seems odd to me because the method used for mutation 
and generationforming are suppose to make it easy to copy find the same top. 

"""
