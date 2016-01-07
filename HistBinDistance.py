import numpy
import csv
import math

### these are the 2 arrays that will supply the values to the hist function
store_array_A = []
store_array_B = []
hist_A = []
hist_B = []
### _JCS is the index of the columns that contain ANY non zero element (>=0.1)
### and _IRS is the row index in the column of that non zero .._SR is the VALUE of the same
_IRS = []
_JCS = []
_SR = []

bin_min = -0.15
bin_max = 0.15
rng = 30
normFactor = 0.9

similarity_matrix = [ [ 0 for i in range(rng+1) ] for j in range(rng+1) ]
### bad hard coding below
store_dailyChg = []
distance_matrix = []
### will return row indices of all non zero indices
### min threshold 0.1
def calcIRS_JCS_SR():
### fill the top half
    for row in range(rng+1):
            for col in range(rng+1):
                if ( col >= row ):
        
                    if ( row == col ):
                        similarity_matrix[row][col] = 1
                    else:
                        similarity_matrix[row][col] = similarity_matrix[row][col-1]*0.5
                        
        ### fill the bottom half
    for col in range(rng+1):
            for row in range(rng+1):
                if ( row >= col ):
        
                    if ( row == col ):
                        similarity_matrix[row][col] = 1
                    else:
                        similarity_matrix[row][col] = similarity_matrix[row-1][col]*0.5         
    
    for col in  range(len(similarity_matrix)):
        
      if ( sum (similarity_matrix[col]) >= 0.1): 
        if len(_IRS) not in _JCS:  
            _JCS.append( len(_IRS) )
        for row in range(len(similarity_matrix)):
          if similarity_matrix[row][col] >= 0.1:
                _IRS.append(row)
                _SR.append(similarity_matrix[row][col])
        if len(_IRS) not in _JCS:      
            _JCS.append( len(_IRS) )

### final entry in _JCS is the number of non empty cells in the similarity matrix i.e. length of _SR
    _JCS.append( len(_SR) )                                    

    
    return 1


### distance calc function
def _calcQCDist(hist_A, hist_B, similarity_matrix, normFactor, sizeOfHist):
    Distance_matrix = []
    sparseInd= 0

    for i in range(sizeOfHist):
        zi= 0.0;
        cb= _JCS[i]
        ce= _JCS[i+1]
        for c in range(cb, ce):
           zi+= (hist_A[_IRS[c]] + hist_B[_IRS[c]])*_SR[sparseInd]
           ++sparseInd

        if (zi!=0.0): 
           
           Distance_matrix.append( (hist_A[i]-hist_B[i])/(pow(zi,normFactor)) )

    
    dist= 0.0
    sparseInd= 0
    
    for i in range(len(Distance_matrix)):
        cb= _JCS[i]
        ce= _JCS[i+1]
        for c in range(cb, ce):
            if( _IRS[c]>=0 and _IRS[c] < len( Distance_matrix )):
                
                dist+= Distance_matrix[i]*Distance_matrix[_IRS[c]]*_SR[sparseInd]
                ++sparseInd;
        

    if dist<0:
            return 0.0
    else:
            return math.sqrt(dist);

    return 1

### read daily change values into array
for colCtr in range(0,7):
    with open('D:\Portfolio-for-masses\DATA\hist_test.csv', 'rb') as csvfile:
        stockReader = csv.reader(csvfile, delimiter=',') 
        locArr = []
        for row in stockReader:
        ### the below is used to check 2 distributions with unequal number of observations come through
           if row[colCtr]!='' :
                locArr.append(float(row[colCtr]))
                
    store_dailyChg.append(locArr)            
### similarity matrix is independant of any loop or whatever and need be called only once
### now call the function that populates irs, jcs and sr

calcIRS_JCS_SR(  )                
                                           
for outer in range(len(store_dailyChg)):
    distArr = []
    for inner in range(len(store_dailyChg)):
      if (outer != inner):        
        hist_A = numpy.histogram( store_dailyChg[outer], rng+1 , (bin_min, bin_max) )        
        hist_B = numpy.histogram( store_dailyChg[inner], rng+1 , (bin_min, bin_max) )        
        
        ### should be passing the 0th element of hist's since the hist is 2 element list that
        ### contains final freq dist and the bin edges
        
        distArr.append( _calcQCDist(hist_A[0], hist_B[0], similarity_matrix, normFactor, len(hist_A[0])) )

      else:
        distArr.append(0)    
    distance_matrix.append(distArr)    


w, v = numpy.linalg.eig( distance_matrix )

print w
print v[5]
