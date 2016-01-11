### -----------
### this file contains routines to extract data from the DB for all MF's based on PHASE and then
### perform clustering and stat analysis. I think bith stat and hist should be different classes all together
### ----------

import django 
import numpy
### some django bull shit
django.setup()

from polls.models import CrisilDbJoin
import numpy as np
from sklearn.cluster import KMeans
from polls.PyFiles.MLModule import MLModule 
from polls.PyFiles.StatModule import StatDo

class _DBExtractChg:
### global datasets
	phase_dt_list = []
	daily_chg_list = dict() 
	master_mf_data_list = dict() 
	returns_list = dict()

	### the below function stores the phase wise retrieval of objects 
	def _getPhaseWiseData(self, locPhase ):
		locMfName = ''
		locArr = []
		dtArr = []
		if locPhase!= '':
			locObj = CrisilDbJoin.objects.filter(phase__contains=locPhase)
			print ' len of locObj '+str(len(locObj))
		else:
			locObj = CrisilDbJoin.objects.all()
		for idx, obj in enumerate (locObj):
			if( locMfName!=obj.mf_nm and len(locArr)>0 ):
				self.master_mf_data_list[locMfName] = locArr
				locArr = []
				locMfName = obj.mf_nm
			elif ( locMfName!=obj.mf_nm and len(locArr) == 0 ):
				locMfName = obj.mf_nm
				locArr.append(obj)				
			else:
				locArr.append(obj)
				dtArr.append(obj.dt)
		return sorted(set(dtArr))	

	### for instance if 2 lists have 1st, 2nd and 4th AND 1,2,3,4 the former should also get 3 with the value of 2/4
	def _calcDailyChg(self, phase_dt_list):
		daily_chg_list = dict()
		ctr = 0
		returns_list = dict()
		for mf, objArr in self.master_mf_data_list.items():
			chgArr = []
			arrCtr = 0
			### the first change% is always going to be 0	
			chgArr.append(0.0)
			### calc returns for the MF for the phase
			if objArr[0].nav > 0 and objArr[len(objArr)-1].nav > 0:
				self.returns_list[mf] = numpy.log( objArr[len(objArr)-1].nav/objArr[0].nav )
			else:
				self.returns_list[mf] = 0.0	
			### the below works like this ..if both st and end dates are equal then update obj ctr and calc the % chg
			### if st dts are equal and end dates aren't then you know end date's ahead ..so don't update obj ctr  
			for idx in range(0, len(phase_dt_list)-1):
			### since objArr is being indexed by +1 as well we need to check if len - 1 has been reached
			    if(arrCtr < len(objArr)-1):
				if objArr[arrCtr].dt == phase_dt_list[idx] and objArr[arrCtr+1].dt == phase_dt_list[idx+1]:
				   if( objArr[arrCtr].nav > 0):	
					chgArr.append( numpy.log( objArr[arrCtr+1].nav/objArr[arrCtr].nav ) )
				   else:
					chgArr.append(0.0)	
					#print 'ok1 '+str(phase_dt_list[idx])+' '+str(phase_dt_list[idx+1])+' '+str(objArr[arrCtr].dt)+' '+str(objArr[arrCtr+1].dt)
					arrCtr = arrCtr + 1
				elif (objArr[arrCtr].dt == phase_dt_list[idx] and objArr[arrCtr+1].dt != phase_dt_list[idx+1]) or (objArr[arrCtr].dt != phase_dt_list[idx] and objArr[arrCtr+1].dt != phase_dt_list[idx+1]):
					chgArr.append(0.0)
					#print 'ok2 '+str(phase_dt_list[idx])+' '+str(phase_dt_list[idx+1])+' '+str(objArr[arrCtr].dt)+' '+str(objArr[arrCtr+1].dt)
				elif (objArr[arrCtr].dt != phase_dt_list[idx] and objArr[arrCtr+1].dt == phase_dt_list[idx+1]):
					chgArr.append(0.0)
					#print 'ok3 '+str(phase_dt_list[idx])+' '+str(phase_dt_list[idx+1])+' '+str(objArr[arrCtr].dt)+' '+str(objArr[arrCtr+1].dt)
					arrCtr = arrCtr + 1
			### we want all arrays of daily returns to be equal to the length of uniqe date list for the range
			### unless arrays are of equal length we can't do a lot of ops on them
			if( len(chgArr) == len(phase_dt_list) ):	
				self.daily_chg_list[mf] = chgArr
			else:
				for looper in range( len(phase_dt_list) - len(chgArr)):
					chgArr.append(0.0)
				self.daily_chg_list[mf] = chgArr
		### clear the mdf data list for the next phase run	
		master_mf_data_list = dict()	
		return 1

	def _retNonZero(self, arr, topOrBottom):
	
		if topOrBottom == 'TOP':
		    looper = 0
		    while  looper < len(arr):
			if arr[looper].nav > 0: 
			   return arr[looper].nav
			   print ' TOP '+str(looper)	 
			   looper = looper + 1

                if topOrBottom == 'BOTTOM':
                    looper = len(arr)-1
                    while  looper > 0:
                        if arr[looper].nav > 0: 
			   return arr[looper].nav 
			   print ' BOTTOM '+str(looper)	
                           looper = looper - 1
		return -1

	def _retDailyChg(self):
		return self.daily_chg_list

### main 
dbInst = _DBExtractChg()
dbInst._calcDailyChg( dbInst._getPhaseWiseData('FLAT1') )
	#print 'mf master len '+str(len(master_mf_data_list))+' daily chg '+str(len(daily_chg_list))
### call clustering
### instantiate ML Module class
_mlMod = MLModule()
for key, val in (_mlMod.clusterInput( dbInst.daily_chg_list )).items():
        print 'Phase '+'FLAT1'+' MF :'+key+','+'Cluster ID '+str(val)+','+str(dbInst.returns_list[key])

### inst Stat Module Class
_statMod = StatDo()
_statMod._calcIRS_JCS_SR()

for mfOuter, llOuter in dbInst.daily_chg_list.items():
	distArr = []
	for mfInner, llInner in dbInst.daily_chg_list.items():
	  if ( mfOuter != mfInner ):
		hist_A = numpy.histogram( llOuter, _statMod.rng+1 , (_statMod.bin_min, _statMod.bin_max) )
		hist_B = numpy.histogram( llInner, _statMod.rng+1 , (_statMod.bin_min, _statMod.bin_max) )	

		distArr.append( _statMod._calcQCDist(hist_A[0], hist_B[0], len(hist_A[0])) )
	  else:
		distArr.append(0)
	
	_statMod.distance_matrix.append(distArr) 
	

