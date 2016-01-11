import numpy
import math

class StatDo:
    ### these are the 2 arrays that will supply the values to the hist function
    store_array_A = []
    store_array_B = []
    hist_A = []
    hist_B = []
    ### _JCS is the index of the columns that contain ANY non zero element (>=0.5)
    ### and _IRS is the row index in the column of that non zero .._SR is the VALUE of the same
    _IRS = []
    _JCS = []
    _SR = []
    
    bin_min = -0.15
    bin_max = 0.15
    rng = 30
    normFactor = 0.9
    _threshold = 0.5   	
    similarity_matrix = [ [ 0 for i in range(rng+1) ] for j in range(rng+1) ]
    store_dailyChg = []
    distance_matrix = []
### will return row indices of all non zero indices
### min threshold defined in variable 
    def _calcIRS_JCS_SR(self):
	### fill the top half
	    for row in range(self.rng+1):
		    for col in range(self.rng+1):
			if ( col >= row ):
		
			    if ( row == col ):
				self.similarity_matrix[row][col] = 1
			    else:
				self.similarity_matrix[row][col] = self.similarity_matrix[row][col-1]*0.5
				
		### fill the bottom half
	    for col in range(self.rng+1):
		    for row in range(self.rng+1):
			if ( row >= col ):
		
			    if ( row == col ):
				self.similarity_matrix[row][col] = 1
			    else:
				self.similarity_matrix[row][col] = self.similarity_matrix[row-1][col]*0.5         
	    
	    for col in  range(len(self.similarity_matrix)):
		
	      if ( sum (self.similarity_matrix[col]) >= self._threshold): 
		if len(self._IRS) not in self._JCS:  
		    self._JCS.append( len(self._IRS) )
		for row in range(len(self.similarity_matrix)):
		  if self.similarity_matrix[row][col] >= self._threshold:
			self._IRS.append(row)
			self._SR.append(self.similarity_matrix[row][col])
		if len(self._IRS) not in self._JCS:      
		    self._JCS.append( len(self._IRS) )

	### final entry in _JCS is the number of non empty cells in the similarity matrix i.e. length of _SR
	    self._JCS.append( len(self._SR) )                                    

	    
	    return 1
	### distance calc function
    def _calcQCDist(self, hist_A, hist_B,  sizeOfHist):
	    Distance_matrix = []
	    sparseInd= 0

	    for i in range(sizeOfHist):
		zi= 0.0;
		cb= self._JCS[i]
		ce= self._JCS[i+1]
		for c in range(cb, ce):
		   zi+= (hist_A[self._IRS[c]] + hist_B[self._IRS[c]])*self._SR[sparseInd]
		   ++sparseInd

		if (zi!=0.0): 
		   
		   Distance_matrix.append( (hist_A[i]-hist_B[i])/(pow(zi,self.normFactor)) )

	    
	    dist= 0.0
	    sparseInd= 0
	    
	    for i in range(len(Distance_matrix)):
		cb= self._JCS[i]
		ce= self._JCS[i+1]
		for c in range(cb, ce):
		    if( self._IRS[c]>=0 and self._IRS[c] < len( Distance_matrix )):
			
			dist+= Distance_matrix[i]*Distance_matrix[self._IRS[c]]*self._SR[sparseInd]
			++sparseInd;
		

	    if dist<0:
		    return 0.0
	    else:
		    return math.sqrt(dist);

	    return 1

