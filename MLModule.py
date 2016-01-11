### -- This class will contain methods for ML ..for now it ll prob just have clustering ----

import numpy as np
from sklearn.cluster import KMeans

class MLModule:
### for now hard code ..later needs to optimize 
   numClusters = 5		
   def clusterInput(self, inputDict):
		print 'vik '+str(len(inputDict))
		localDict = dict()
		localList = []
		returnDict = dict()
		indx = 0
		for key, val in inputDict.items():
			localDict[indx] = key
			indx = indx + 1	
			localList.append(val)

		yPred = KMeans( n_clusters = self.numClusters).fit_predict(localList) 	
		for idx in range(len(yPred)):
			returnDict[localDict[idx]] = yPred[idx]		
                
                return returnDict		
