import numpy as np
import math

class input:
    def __init__(self, value, name = "Input"):
        """Initializes the input and name of input class"""
        self.value = value 
        self.name = name

    def __str__(self, input):
        """Return input name and value of the input"""
        return input + self.name +" value: " + str(self.value)

    def getValue(self):
        """Returns the value of the input"""
        return self.value

    def setValue(self, value):
        """Sets value of input"""
        self.value = value

class pNeuron:
    def __init__(self, threshold, inputs):
        """Initializes the input and 'threshold of p(perceptron)Neuron class"""
        self.threshold = threshold
        self.inputs = inputs 

    def __str__(self, input = ""):
        """Return string with tree of neuron and its inputs"""
        if(input == ""):
            string = "Neuron:\nThreshold: " + str(self.threshold) + "\nInput: " + "\n\t" + "========================"
        else:
            string = input + "Neuron:\n" + input + "Threshold:"+ str(self.threshold) + "\n"+ input + "Input" + "\n" + input + "\t"  +"========================"
        
        for i, j in self.inputs:
            string += "\n"
            string += i.__str__(input + "\t") + "\tWeight:" + str(j)
        string += "\n" + input + "\t" + "========================\n"
        return string
    

    def execute(self):
        """Returns whether the inputs combine to be higher or equal to the threshold"""
        total = 0
        for i,j in self.inputs:
            #j is the weight
            total += (i.getValue() * j) 
        return int(total >= self.threshold)

    def addInput(self, input):
        """Adds input to list with inputs"""
        self.inputs += input
          
    def getValue(self):
        """Executes neuron, used for recursive calls in execute"""
        return self.execute()


class neuron:
    def __init__(self, inputs, name = "Generic"):
        """Initializes the class with inputs and a name"""
        self.inputs = inputs
        self.learningRate = 0.1
        self.name = name

    def __str__(self, input = ""):
        """Return string with tree of neuron and its inputs"""
        if(input == ""):
            string = self.name + " :\n" + "\nInput: " + "\n\t" + "========================"
        else:
            string = input + self.name + " :\n" + input + "Input" + "\n" + input + "\t"  +"========================"
        
        for i, j in self.inputs:
            string += "\n"
            string += i.__str__(input + "\t") + "\tWeight:" + str(j)
        string += "\n" + input + "\t" + "========================\n"
        return string

    def update(self, y):
        """Updates the weights with y as the expected value"""
        currentvalue = 0
        if(y != 0 or y != 1):
            currentvalue = y
        else:
            currentvalue = self.execute() - y

        if(self.execute() == y):
            return "Learning complete!"
        k = 0
        for i, j in self.inputs:
            newWeight = j + self.learningRate * i.getValue() * (1 - math.tanh(i.getValue())) * (currentvalue)
            #Apparently python doesn't allow for tuple values to be changed. 
            #That is why all this casting is needed for this to work. 
            newTuple = list(self.inputs[k])
            newTuple[1] = newWeight
            self.inputs[k] = tuple(newTuple) 
            k += 1
        
    def backPropogation(self, deltaOutput):
        """Executes backprpogation on neuron"""
        self.update(deltaOutput)
        for i,j in self.inputs:
            total = (1 - i.getValue()) * j * deltaOutput
            if(type(i).__name__ == 'neuron'):
                i.backPropogation(deltaOutput)
                i.update(total)
            elif(type(i).__name__ == 'input'):
                pass
        pass 

    def getWeights(self):
        """Returns a string with the weights """
        returnStr = ""
        for i, j in self.inputs:
            j = round(j, 2)
            returnStr += str(j)
            returnStr += "\t"

        return returnStr
    
    def execute(self):
        """Return the output from the neuron """
        total = 0
        for i,j in self.inputs:
            total += (i.getValue() * j)
        #print(total)
        return math.tanh(total)

    def getValue(self):
        """Executes neuron, used for recursive calls in execute"""
        return self.execute()

    def calculateOutputDelta(self, requiredValue, error = False):
        """Calculates and returns the error of the current neuron, when error is false the neuron is viewed as an output neuron"""
        total = 0
        for i, j in self.inputs:
            total += i.getValue()* j
        if(error):
            return (1 - (math.tanh(math.tanh(total)))) * requiredValue
        else:
            return (1 - (math.tanh(math.tanh(total)))) * (requiredValue - self.execute()) 


from random import shuffle
import random

class network:
    def __init__(self, inputs, outputs, otherLayers = []):
        """Initializes network with the inputs, outputs and otherlayers"""
        self.inputs = inputs
        self.outputs = outputs
        self.otherLayers = otherLayers

    def execute(self, inputValues):
        """Executes network and returns list of tuples containing name of output and value"""
        error= []
        if (len(inputValues) < len(self.inputs)):
            return None
        k = 0
        for i in self.inputs:
            i.setValue(inputValues[k])
            k+= 1

        k = 0
        for i in self.outputs:
            error.append((i.execute(), i.name))
            k+= 1
        return error


    def test(self, inputValues, expectedOutputValues):
        """Executes network and returns list of tuples containing name of output and error of that output"""
        error = []
        if (len(inputValues) < len(self.inputs)):
            return None
        k = 0
        for i in self.inputs:
            i.setValue(inputValues[k])
            k+= 1

        k = 0
        for i in self.outputs:
            error.append((expectedOutputValues[k] - i.execute(), i.name))
            k+= 1

        return error

    def train(self, inputValues, expectedOutputValues, amount):
        """Sets input of network and executes backprpogation to alter the weights"""
        k = 0
        for j in self.inputs:
            j.setValue(inputValues[k])
            k+= 1
        for i in range(0, amount):
            #amount of trainingf
            self.backpropogation(expectedOutputValues)
        pass 




    def backpropogation(self, expectedOutputValues):
        """Executes backpopogation on whole network given the following expected output values""" 
        deltaDifference = []
        for i in self.outputs:
            deltaDifference.append(i.calculateOutputDelta(expectedOutputValues[self.outputs.index(i)]))
       
        for h in range(0, len(self.outputs)):
            self.outputs[h].update(deltaDifference[h])
            #Eerste laag berekent

            #start.update(start.calculateOutputDelta())

        #this is the sum of all errors
        #first part of the formula
        # to 
        lastLayer = self.outputs
        temp = []
        #lastLayer is used for the output
        for layer in self.otherLayers:
            for _neuron in layer:
                sum = 0
                for output in lastLayer:
                    #Compare the neuron to see if the same if yes, calculate weight
                    for h, w in output.inputs:
                        if(type(h).__name__ == 'input'):
                            #print("input layer found")
                            return "Succes"
                        if h == _neuron:
                            sum += w * deltaDifference[lastLayer.index(output)]
                #Now you have te sum for the specific neuron
                #print(_neuron)
                newDelta = _neuron.calculateOutputDelta(sum, True)
                temp.append(newDelta)
                #Now temp consist of the error from the next layer
                #Now use this to calcute new weights 
                _neuron.update(newDelta)
            deltaDiffence = temp
            lastLayer = layer


            #1. Bereken gewichten met error

             

        pass






#Excersise D: Iris dataset



#Excersise 1: NOR-Gate

inputs = [(input(0), 0.5), (input(0), 0.5), (input(0), 0.5)]
orGate = pNeuron(0.5, inputs)
notInputs = [(orGate, -1)]
norGate = pNeuron(-.5, notInputs)


print("Nor-Gate")
print("A\tB\tC\tOutput")
print("0\t0\t0\t", norGate.execute())

inputs[2][0].setValue(1)
print("0\t0\t1\t", norGate.execute())

inputs[2][0].setValue(0)
inputs[1][0].setValue(1)
print("0\t1\t0\t", norGate.execute())

inputs[0][0].setValue(1)
inputs[1][0].setValue(0)
print("1\t0\t0\t", norGate.execute())

inputs[2][0].setValue(1)
inputs[1][0].setValue(1)
inputs[0][0].setValue(1)
print("1\t1\t1\t", norGate.execute())

#Exercise 2: Neural ADDER

inputsNand = [(input(0),-1), (input(0), -1)]
nand = pNeuron(-1.5, inputsNand)

print("\nNand-Gate")
print("A\tB\tOutput")
print("0\t0\t", nand.execute())
inputsNand[0][0].setValue(1)
print("1\t0\t", nand.execute())
inputsNand[0][0].setValue(0)
inputsNand[1][0].setValue(1)
print("0\t1\t", nand.execute())
inputsNand[0][0].setValue(1)
print("1\t1\t", nand.execute())

inputA = input(0)
inputB = input(0)

nand1 = pNeuron(-1.5, [(inputA,-1), (inputB, -1)])
nand2 = pNeuron(-1.5, [(nand1, -1), (inputA, -1)])
nand3 = pNeuron(-1.5, [(nand1, -1), (inputB, -1)])
carryNand = pNeuron(-1.5, [(nand1, -1), (nand1, -1)])
sumNand = pNeuron(-1.5, [(nand2, -1), (nand3, -1)])

print("\nAdder:")
print("A\t B\tCarry\tSum")
print(inputA.getValue(),"\t",inputB.getValue(), "\t", carryNand.execute(), "\t", sumNand.execute())
inputA.setValue(1)
print(inputA.getValue(),"\t",inputB.getValue(), "\t", carryNand.execute(), "\t", sumNand.execute())
inputA.setValue(0)
inputB.setValue(1)
print(inputA.getValue(),"\t",inputB.getValue(), "\t", carryNand.execute(), "\t", sumNand.execute())
inputA.setValue(1)
inputB.setValue(1)
print(inputA.getValue(),"\t",inputB.getValue(), "\t", carryNand.execute(), "\t", sumNand.execute())

#XOR backpropogation

i = input(1, "I")
j = input(1, "J")
h = neuron([(i, 0.2), (j, -0.4)], "H")
l = neuron([(i, 0.7), (j, 0.1)], "L")
k = neuron([(h,0.5)], "K")
g = neuron([(l,0.2)], "G")

o = neuron([(k, 0.6), (g, 0.9)])

print("Exercise XOR")

print("Weights neural network before backprogation")
print(o.getWeights())
print(h.getWeights())
print(l.getWeights())
print(k.getWeights())
print(g.getWeights())

o.backPropogation(o.calculateOutputDelta(1))
print()

print("Weights neural network after backprogation")
print(o.getWeights())
print(h.getWeights())
print(l.getWeights())
print(k.getWeights())
print(g.getWeights())

#print("")
print(o)



#using example from: https://www.neuraldesigner.com/learning/examples/iris_flowers_classification
"""
I used this structure because that way I could follow the tutorial. I wanted to try out some more structure but didn't have the time
"""

#initializing inputs
sepal_length = input(0, "sepal_length")
sepal_width = input(0, "sepal_width")
petal_length = input(0, "petal_length")
petal_width = input(0, "petal_width")

#initializing first layer with random weights
# This is called the scaling layer
Layer1A = neuron([(sepal_length, random.uniform(-0.1, 0.1)), (sepal_width, random.uniform(-0.1, 0.1)), (petal_length, random.uniform(-0.1, 0.1)), (petal_width, random.uniform(-0.1, 0.1))], "Layer 1 Neuron 1") 
Layer1B = neuron([(sepal_length, random.uniform(-0.1, 0.1)), (sepal_width, random.uniform(-0.1, 0.1)), (petal_length, random.uniform(-0.1, 0.1)), (petal_width, random.uniform(-0.1, 0.1))], "Layer 1 Neuron 2") 
Layer1C = neuron([(sepal_length, random.uniform(-0.1, 0.1)), (sepal_width, random.uniform(-0.1, 0.1)), (petal_length, random.uniform(-0.1, 0.1)), (petal_width, random.uniform(-0.1, 0.1))], "Layer 1 Neuron 3") 
Layer1D = neuron([(sepal_length, random.uniform(-0.1, 0.1)), (sepal_width, random.uniform(-0.1, 0.1)), (petal_length, random.uniform(-0.1, 0.1)), (petal_width, random.uniform(-0.1, 0.1))], "Layer 1 Neuron 4")
Layer1E = neuron([(sepal_length, random.uniform(-0.1, 0.1)), (sepal_width, random.uniform(-0.1, 0.1)), (petal_length, random.uniform(-0.1, 0.1)), (petal_width, -random.uniform(-0.1, 0.1))], "Layer 1 Neuron 5") 

#initializing second layer with random weights 
Layer2A = neuron([(Layer1A, random.uniform(-0.1, 0.1)), (Layer1B, random.uniform(-0.1, 0.1)), (Layer1C, random.uniform(-0.1, 0.1)), (Layer1D, random.uniform(-0.1, 0.1)), (Layer1E, random.uniform(-0.1, 0.1))], "Layer 2 Neuron 1")
Layer2B = neuron([(Layer1A, random.uniform(-0.1, 0.1)), (Layer1B, random.uniform(-0.1, 0.1)), (Layer1C, random.uniform(-0.1, 0.1)), (Layer1D, random.uniform(-0.1, 0.1)), (Layer1E, random.uniform(-0.1, 0.1))], "Layer 2 Neuron 2")
Layer2C = neuron([(Layer1A, random.uniform(-0.1, 0.1)), (Layer1B, random.uniform(-0.1, 0.1)), (Layer1C, random.uniform(-0.1, 0.1)), (Layer1D, random.uniform(-0.1, 0.1)), (Layer1E, random.uniform(-0.1, 0.1))], "Layer 2 Neuron 3")



#initializing the output layer with random weights
iris_setosa = neuron([(Layer2A, random.uniform(-0.1, 0.1)), (Layer2B, random.uniform(-0.1, 0.1)), (Layer2C, random.uniform(-0.1, 0.1))], "Iris setosa" )
iris_versicolor = neuron([(Layer2A, random.uniform(-0.1, 0.1)), (Layer2B, random.uniform(-0.1, 0.1)), (Layer2C, random.uniform(-0.1, 0.1))], "Iris versicolor" )
iris_virginica = neuron([(Layer2A, random.uniform(-0.1, 0.1)), (Layer2B, random.uniform(-0.1, 0.1)), (Layer2C, random.uniform(-0.1, 0.1))], "Iris virginica" )

layer1 = [Layer1A, Layer1B, Layer1C, Layer1D, Layer1E]
layer2 = [Layer2A, Layer2B, Layer2C]

iris = network([sepal_length, sepal_width, petal_length, petal_width], [iris_setosa, iris_versicolor, iris_virginica], [layer2, layer1])

print(iris.test([5.1, 3.5, 1.4, 0.2], [1,0,0]))
iris.backpropogation([1,0,0])
#The dataset


text_file = open("iris.data", "r")
lines = text_file.read()
test = lines.split('\n')
#last two lines are spaces so these need to be removed 
test.pop()
test.pop()


def dataParser(dataline):
    if(dataline == ''):
        print("No data given")
        return None

    differentValues = dataline.split(',')
    #got different values in string format
    #the first four values should be converted to floats (the fact that python just fixes this for me without any problems just made my day)
    inputValues = []
    for i in range(0,4):
        inputValues.append(float(differentValues[i])) 
    if(differentValues[4] == "Iris-virginica"):
        #is the last one
        return [inputValues, [0,0,1]]
    elif(differentValues[4] == "Iris-versicolor"):
        return [inputValues, [0,1,0]]
    elif(differentValues[4] == "Iris-setosa"):
        return [inputValues, [1,0,0]]

    return None


def validation(validationList):
    correct = 0
    for i in range(0, len(validationList)):
        values = dataParser(validationList[i])
        new = []
        _new = [0,0,0]
        if(values != None):
            list = iris.execute(values[0])
            #print(list)
            for i, j in list:
                #i is error
                new.append(i)
            _new[new.index(max(new))] = 1
            #print(_new)
            if(_new == values[1]):
                correct += 1

    return(correct)
                            



shuffle(test)
testpercentage = int(len(test) / 4)
testSet = test[:testpercentage]
validationSet = test[testpercentage:]

while True:
    for i in range(0, len(testSet)):
        values = dataParser(test[i])
        iris.train(values[0],values[1], 1)
    
   
    print("Amount right: ", validation(validationSet))

