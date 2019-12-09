import keras.backend as K
def rob_loss(y_true,y_pred):
    return K.sum( K.abs(y_true-y_pred))/K.abs(K.mean(y_true)-K.mean(y_pred))

def rob_metric(y_true, y_pred):
    return K.mean(K.max(y_true)-K.max(y_pred))
    
    
    
    
    
    
    
    
    
from numpy import array as ar
from numpy import exp as e
import pickle
def dumper(a,where):
	pickle.dump(a,open(where,'wb'))

def find_percentage(A,B):
	seed=0
	for i in range(0,len(A)):
		if A[i][B[i]]==max(A[i]):
			seed=seed+1
	return seed/len(A)

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_answer(L):
	LO=[]
	for i in range(0,len(L)):
		LO.append(L[i][0])
	return LO

def normalize(s):
	m=[]
	ret=[]
	for l in range(0,len(s)):
		m.append(max(s[l]))
		ret.append([])
		for j in range(0,len(s[l])):
			ret[l].append(s[l][j]/m[l])
	ret=ar(ret)
	for i in range(0,len(ret)):
		ret[i]=ar(ret[i])	
	return ret
	
def softmax(logit,t):
	summer=[]
	answer=[]
	for l in range(0,len(logit)):
		summer.append(0)
		for j in range(0,len(logit[l])):
			summer[l]=summer[l]+e(logit[l][j]/t)
	for z in range(0,len(logit)):
		answer.append([])
		for j in range(0,len(logit[z])):
			answer[z].append(e(logit[z][j]/t)/summer[z])
	answer=ar(answer)
	for i in range(0,len(answer)):
		answer[i]=ar(answer[i])
	return answer
	
	
	
def Rob(s,L,r,p,sig,sof):
	c=[]
	Q=[]
	R=[]
	
	for l in range(0,len(s)):
		c.append(s[l][L[l]])

	for l in range(0,len(s)):
		Q.append([])
		for j in range(0,len(s[l])):
			Q[l].append(abs(c[l]-s[l][j]))
	allQ=[]
	for l in range(0,len(s)):
		allQ.append(0)
		for j in range(0,len(s[l])):
			allQ[l]=allQ[l]+Q[l][j]
	for l in range(0,len(s)):
		allQ[l]=allQ[l]/len(s[l])
	for l in range(0,len(s)):
		R.append([])
		M=max(s[l])
		for j in range(0,len(s[l])):
			if Q[l][j]<=sig*allQ[l] and j!=L[l]:
				R[l].append((r*sof)+s[l][j])
			if Q[l][j]>sig*allQ[l] and j!=L[l]:
				R[l].append(s[l][j]-(p*Q[l][j]))
			if j==L[l]:
				R[l].append(M+r)
	R=ar(R)
	for i in range(0,len(R)):
		R[i]=ar(R[i])
	
	#print(len(R))
	return R
	
	
	
	
	


