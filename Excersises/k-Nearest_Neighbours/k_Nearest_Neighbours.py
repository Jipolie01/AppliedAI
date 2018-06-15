import numpy as np
import operator
import time
import math

class kNN:
    def __init__(self):
        self.validation = np.genfromtxt('validation1.csv', delimiter = ';', usecols = [1,2,3,4,5,6,7], 
                            converters= { 5: lambda s: 0 if s == b"-1" else float(s),
                                         7: lambda s: 0 if s == b"-1" else float(s)})
        self.validationLabels = []
        self.dataset = np.genfromtxt('dataset1.csv', delimiter = ';', usecols = [1,2,3,4,5,6,7], 
                     converters= { 5: lambda s: 0 if s == b"-1" else float(s),
                                  7: lambda s: 0 if s == b"-1" else float(s)})
        self.datalabels = []
        self.days = np.genfromtxt('days.csv', delimiter = ';', usecols = [1,2,3,4,5,6,7], 
                     converters= { 5: lambda s: 0 if s == b"-1" else float(s),
                                  7: lambda s: 0 if s == b"-1" else float(s)})
        self.makeDateTabels()

    def makeDateTabels(self):
        dates = np.genfromtxt('dataset1.csv', delimiter = ';', usecols = [0])
        for label_1 in dates:
            if label_1 < 20000301:
                self.datalabels.append('winter')
            elif 20000301 <= label_1 < 20000601:
                self.datalabels.append('lente')
            elif 20000601 <= label_1 < 20000901:
                self.datalabels.append('zomer')
            elif 20000901 <= label_1 < 20001201:
                self.datalabels.append('herfst')
            else:
                self.datalabels.append('winter')
        validation_dates = np.genfromtxt('validation1.csv', delimiter = ';', usecols = [0])
        for label_1 in validation_dates:
            if label_1 < 20010301:
                self.validationLabels.append('winter')
            elif 20010301 <= label_1 < 20010601:
                self.validationLabels.append('lente')
            elif 20010601 <= label_1 < 20010901:
                self.validationLabels.append('zomer')
            elif 20010901 <= label_1 < 20011201:
                self.validationLabels.append('herfst')
            else:
                self.validationLabels.append('winter')

    def calculateDistance(self, pointA, pointB):
            return np.linalg.norm(pointA - pointB)

    def calculateClosest(self, pointA, restOfData, k):
            closestDistance = [];
            closestDistanceIndex = [];
            index = 0
            for column in restOfData:
                distance = self.calculateDistance(pointA, column)

                if len(closestDistance) < k:
                    closestDistance.append(distance)
                    closestDistanceIndex.append(index)

                elif distance < max(closestDistance) and distance != 0 :
                    closestDistanceIndex[closestDistance.index(max(closestDistance))] = index
                    closestDistance[closestDistance.index(max(closestDistance))] = distance
                    
                #print(distance, " : " , closestDistance, " : ", index, " : ", closestDistanceIndex)
                index += 1
            seasons =  {'zomer' : 0, 'herfst': 0, 'winter' : 0,'lente': 0} 
            for i in closestDistanceIndex:
                if self.datalabels[i] == 'herfst':
                    seasons['herfst'] += 1
                elif self.datalabels[i] == 'winter':
                    seasons['winter'] += 1
                elif self.datalabels[i] == 'lente':
                    seasons['lente'] += 1
                elif self.datalabels[i] == 'zomer':
                    seasons['zomer'] += 1

            return max(seasons.items(), key=operator.itemgetter(1))[0]

    def testK(self):
            index_1 = 0
            matrix = []
            bestK = 0
            lowestRight = 0
            for k in range(1,101):
                index_1 = 0
                matrix = []
                bestK = 0
                lowestRight = 0
                for point in self.validation:
                    index = self.calculateClosest(point, self.dataset , k)
                    matrix.append([self.validation[index_1], index, self.validationLabels[index_1]])
                    index_1 += 1

                right = 0
                total = 0
                for i in matrix:
                    if i[1] == self.validationLabels[total]:
                        right += 1
                    total += 1
                print("k :  ", k, "\t total right: ", right, " : ", total)

                if(right < lowestRight):
                    lowestRight = right
                    bestK = k
                index_1 = 0
                total = 0
            return bestK

    def printSeasons(self,k):
            matrix = []
            index = 0
            for point in self.days:
    
                season = self.calculateClosest(point,self.dataset, 43)
                matrix.append([self.days[index], season])
                index += 1

            for i in matrix:
                print(i)


h = kNN()
h.testK()

