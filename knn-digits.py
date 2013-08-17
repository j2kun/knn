import knn
import random

def column(A, j):
   return [row[j] for row in A]

def digitsData():
   ''' Read in the handwritten digits data from the file 'digits.dat', and
       return the data points and their labels as two lists. '''
   with open('digits.dat') as inFile:
      lines = inFile.readlines()

   data = [line.strip().split(',') for line in lines]
   data = [([int(x) for x in point.split()], int(label)) for (point, label) in data]

   return data

def test(data, k):
   random.shuffle(data)
   pts, labels = column(data, 0), column(data, 1)

   trainingData = pts[:800]
   trainingLabels = labels[:800]
   testData = pts[800:]
   testLabels = labels[800:]

   f = knn.makeKNNClassifier(trainingData, trainingLabels, k, knn.euclideanDistance)
   correct = 0.0
   total = len(testLabels)

   for (point, label) in zip(testData, testLabels):
      if f(point) == label:
         correct += 1

   return correct/total


if __name__ == "__main__":
   data = digitsData()
   print "k\tcorrect"

   for k in range(16,50):
      successRate = test(data, k)
      print "%d\t%.3f" % (k, successRate)

