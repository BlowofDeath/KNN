import math
import numpy as np
from csv import reader

def groupingDistance(A):
    temp = []
    decisions = np.unique(A[:,A[1].size-1])
    for i in decisions:
        helper = A[i==A[:,A[1].size-1]]
        temp.append(helper[:,:helper[1].size-1])
    
    return [temp, decisions]
           

def euclideanDistance(sample_one, sample_two):
    distance =0
    for i in range(len(sample_one)-1):
        distance+= pow(sample_one[i] - sample_two[i],2)
    return math.sqrt(distance)

def groupingDistanceToX(A, x):
    data = groupingDistance(A)
    groups = {}
    for j in range(len(data[0])):
        temp = []
        for i in data[0][j]:
            temp.append(euclideanDistance(i, x)) 
        groups.update({data[1][j]: temp})
    return groups
def Knn(A, k, x):
    A = groupingDistanceToX(A, x)
    groups = {}
    for j in A:
        summary=0
        temp = np.sort(A[j])
    
        for i in range(k):
            # print("j: " + str(temp[i]))
            summary+=temp[i]
        groups.update({j: summary})
    # print(groups)
    # print(min(groups, key=groups.get))
    return min(groups, key=groups.get)

def oneVsRest(A, k):
    
    correct = 0
    for i in range((A.shape[0])):
        temp = A[i]
        T = np.delete(A, i-1, 0)
        resoult = Knn(T,k,temp[:temp.size-1])

        if(resoult == temp[temp.size-1]):
            correct+=1
    print("All = "+str(A.shape[0])+": Correct = "+str(correct))
    accuracy = (100/A.shape[0]) * correct
    print("Accuracy = " +str(accuracy)+"%")
    return accuracy
        
            
            
        
        
    

        
    
    
 

with open('iris.txt', 'r') as csv_data:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(csv_data, delimiter='\t')
    # Pass reader object to list() to get a list of lists
    A = np.array(list(csv_reader)).astype('float64')
  
    #Resoult=Knn(A,4, [5, 8, 8, 4])
    oneVsRest(A,4)
    
    
    








