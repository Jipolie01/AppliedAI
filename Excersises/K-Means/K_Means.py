import numpy as np
import random 
import operator

#kies een random data punt uit data set voor cluster midden
class cluster:
    def __init__(self, middle):
        self.points = []
        self.middle = middle

    def addPoint(self, point):
        self.points.append(list(point))

    def __str__(self):
        return str(self.middle)

    def calculateMiddle(self):
        totalNumber = 0
        counter = 0
        if(len(self.points) > 0):
            for i in self.points:
                i = i[1:]
                totalNumber +=  np.array(i)
                counter += 1
            self.middle = totalNumber/ counter

    def distanceToMiddle(self, point):
        array = np.delete(point, 0)
        middle = self.middle
        if(len(self.middle) != 7):
            middle = np.delete(self.middle, 0)
        return np.linalg.norm(middle - array)

    def removeFromList(self, point):
        for i in self.points:
            if( list(i) == list(point)):
                self.points.remove(list(point))

    def label(self):
        dict = {'herfst': 0, 'winter': 0, 'zomer' : 0, 'lente' : 0}
        for i in self.points:
            if i[0] < 20000301:
                dict['winter'] += 1
            elif 20000301 <= i[0] < 20000601:
                dict['lente'] += 1
            elif 20000601 <= i[0] < 20000901:
                dict['zomer'] += 1
            elif 20000901 <= i[0] < 20001201:
                dict['herfst'] += 1
            else:
                dict['winter'] += 1
        return dict


        

#vanuit de punten kijken voor recalculate niet voor de clusters 


def chooseRandom(amountOfK, dataset):
    points = []
    for i in range(0,amountOfK):
        temp = random.randint(0, len(dataset))
        while temp in points:
            temp = random.randint(0, len(dataset))
        points.append(temp)
    
    randomClusters = []
    for k in points:
        randomClusters.append(cluster(np.delete(dataset[k], 0)))

    #weet limiet van punten
    #pak minimale en maximale punten en kies iets daartussen 
    #geef lijst van clusters terug 
    return randomClusters

def manageClusters(dataset, _dataset):
    #doe algorithme 
    #generate random points
    randomClusters = chooseRandom(3, dataset)
    lowestCluster = None
    for amount in range(0,30):
        string = ""
        for points in dataset:
            lowestDistance = None
            for clusters in randomClusters:
                distance = clusters.distanceToMiddle(points)
                if lowestDistance == None or distance < lowestDistance:
                    lowestDistance = distance
                    lowestCluster = clusters
                clusters.removeFromList(points)
            lowestCluster.addPoint(points)
        for clusters in randomClusters:
            clusters.calculateMiddle()
            dict = clusters.label()
            print(dict)
            string = string + str(len(clusters.points)) + " - Season: " + str(max(dict.items(), key=operator.itemgetter(1))[0]) + " \n"
        print(string," \nRound : ", amount)

dataset = np.genfromtxt('dataset1.csv', delimiter = ';', usecols = [0,1,2,3,4,5,6,7], 
                     converters= { 5: lambda s: 0 if s == b"-1" else float(s),
                                  7: lambda s: 0 if s == b"-1" else float(s)})

_dataset = np.genfromtxt('dataset1.csv', delimiter = ';', usecols = [1,2,3,4,5,6,7], 
                     converters= { 5: lambda s: 0 if s == b"-1" else float(s),
                                  7: lambda s: 0 if s == b"-1" else float(s)})

manageClusters(dataset, _dataset)

#numpy.choose geef array en dan keuzes voor de arrays n