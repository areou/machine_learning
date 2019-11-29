import pickle
from numpy import exp as e
import matplotlib.pyplot as plt


#THIS IS MY THING:
#S=LOGITS
#L=LABELS (0,..,1,0,...,0) I.E. 1 AT CORRECT LABEL 0 EVERYWHERE ELSE
#R=REWARD PARAMETER
#P=PUNISHMENT PARAMETER
#SIG=SIGNIFACENCE PARAMATER
#SOF=SOFTENING PARAMETER
def Rob(s,L,r,p,sig,sof):
	c=[]
	Q=[]
	R=[]
	
	for l in range(0,len(s)):
		c.append(s[l][L[l]])

	for l in range(0,len(s)):
		Q.append([])
		for j in range(0,100):
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
		for j in range(0,len(s[l])):
			if Q[l][j]<=sig*allQ[l] and j!=L[l]:
				R[l].append(r*sof*s[l][j])
			if Q[l][j]>sig*allQ[l] and j!=L[l]:
				R[l].append(s[l][j]-(p*Q[l][j]))
			if j==L[l]:
				R[l].append(r*s[l][j])
	
	#print(len(R))
	return R








#THIS IS THE STUFF I COULDN'T LOAD FROM TENSOR FLOW SO I MADE IT
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
def zeros():
	return [0]*100
	
	
	
def softmax(logit,t):
	summer=[]
	answer=[]
	for l in range(0,len(logit)):
		summer.append(0)
		for j in range(0,100):
			summer[l]=summer[l]+e(logit[l][j]/t)
	for z in range(0,len(logit)):
		answer.append([])
		for j in range(0,100):
			answer[z].append(e(logit[z][j]/t)/summer[z])

	return answer
	
	











#AND THIS STUFF JUST SETS IT UP SO I CAN LOOK AT SOME OF THE ANSWERS
#WHEN YOU DO YOUR STUFF YOU MAY WANT TO NORMALIZE STILL
	
# load teacher logits
filehandler = open("cifar100_10_soft_targets.pkl", 'rb')
logits = pickle.load(filehandler)	

	

labels=[]	
for j in range(0,50000):
	labels.append(zeros())
	



#print(type(labels))
    
poop=unpickle('train')

#print(poop[b'fine_labels'][1])

ans=poop[b'fine_labels']


#print(poop.keys())
for i in range(0,len(poop[b'fine_labels'])):
	for j in range(0,100):
		if poop[b'fine_labels'][i]==j:
			labels[i][j]=1
			


k=int(input('which one? '))

reward=2
punishment=4
significance=.5
soft=.9
temp=15
	 	
forGraphLogit=[logits[k]]
forGraphLabel=[labels[k]]

forGraphSoft=softmax(forGraphLogit,temp)
OG=softmax(forGraphLogit,1)


#for all
#S=softmax(logits)

#print(S[j])

forGraphAns=[ans[k]]




N=Rob(forGraphLogit,forGraphAns,reward,punishment,significance,soft)

M=softmax(N,temp)

x=[]
for i in range(0,100):
	x.append(i)


	
#for all	
#N=Rob(S,labels,1,.5)


# plotting the points
f=plt.figure(1)
plt.plot(x,forGraphSoft[0],label='Soft-max')
  
plt.plot(x, M[0],label='Rob') 

plt.plot(x, OG[0],label='temperature=1')

#plt.plot(x,labels[2],label='Label')


  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 


  
# giving a title to my graph 
plt.title('norm-test: correct answer= '+str(forGraphAns[0])+' temperature= '+str(temp)) 

plt.legend() 


g=plt.figure(2)

plt.plot(x,forGraphLogit[0],label='Original Logits')
  
plt.plot(x, N[0],label='Clean Logits') 

#plt.plot(x,labels[2],label='Label')


  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 


  
# giving a title to my graph 
plt.title('Logits: r= '+str(reward)+' p= '+str(punishment)+' sig= '+str(significance)+' softness= '+str(soft))

plt.legend() 


"""
h=plt.figure(3)
plt.plot(x,forGraphSoft[0],label='Soft-max_base_2')
  
plt.plot(x, forGraphSoftE[0],label='Soft-max_O_G') 

#plt.plot(x,labels[2],label='Label')


  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 


  
# giving a title to my graph 
plt.title('norm-test: correct answer= '+str(forGraphAns[0])+' softner= '+str(soft)) 

plt.legend() 

"""
  
# function to show the plot 
plt.show() 









