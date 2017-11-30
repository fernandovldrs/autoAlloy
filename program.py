import numpy as np
import os.path

factor=[]
N=[]
M=[]

def getBasicOpZB():	
	I= np.array([[1, 0,  0, 0  ],[ 0, 1,  0, 0  ],[ 0, 0,  1, 0  ],[ 0, 0, 0, 1]]); #identity	
	T1=np.array([[1, 0,  0, 0.5],[ 0, 1,  0, 0.5],[ 0, 0,  1, 0  ],[ 0, 0, 0, 1]]);	#translational matrices
	T2=np.array([[1, 0,  0, 0  ],[ 0, 1,  0, 0.5],[ 0, 0,  1, 0.5],[ 0, 0, 0, 1]]);
	T3=np.array([[1, 0,  0, 0.5],[ 0, 1,  0, 0  ],[ 0, 0,  1, 0.5],[ 0, 0, 0, 1]]);	
	R1=np.array([[0, -1, 0, 0  ],[ 1, 0,  0, 0  ],[ 0, 0, -1, 0  ],[ 0, 0, 0, 1]]);	#rotational matrices
	R2=np.array([[1, 0,  0, 0  ],[ 0, -1, 0, 0  ],[ 0, 0, -1, 0  ],[ 0, 0, 0, 1]]);
	R3=np.array([[0, 0,  1, 0  ],[ 1, 0,  0, 0  ],[ 0, 1,  0, 0  ],[ 0, 0, 0, 1]]);
	basicOp=[I, T1, T2, T3, R1, R2, R3]
	return basicOp

#group designation:  186; P 63 m c;  P 6c -2c
def getBasicOpWZ():	
	I= np.array([[1, 0,  0, 0  ],[ 0, 1,  0, 0  ],[ 0, 0,  1, 0  ],[ 0, 0, 0, 1]]); #identity	
	T1=np.array([[1, 0,  0, 0.5],[ 0, 1,  0, 0.5],[ 0, 0,  1, 0  ],[ 0, 0, 0, 1]]);	#translational matrices
	T2=np.array([[1, 0,  0, 0  ],[ 0, 1,  0, 0.5],[ 0, 0,  1, 0.5],[ 0, 0, 0, 1]]);
	T3=np.array([[1, 0,  0, 0.5],[ 0, 1,  0, 0  ],[ 0, 0,  1, 0.5],[ 0, 0, 0, 1]]);	
	R1=np.array([[0, -1, 0, 0  ],[ 1, 0,  0, 0  ],[ 0, 0, -1, 0  ],[ 0, 0, 0, 1]]);	#rotational matrices
	R2=np.array([[1, 0,  0, 0  ],[ 0, -1, 0, 0  ],[ 0, 0, -1, 0  ],[ 0, 0, 0, 1]]);
	R3=np.array([[0, 0,  1, 0  ],[ 1, 0,  0, 0  ],[ 0, 1,  0, 0  ],[ 0, 0, 0, 1]]);
	basicOp=[I, T1, T2, T3, R1, R2, R3]
	return basicOp

def getPrimBasisZB():
	basis=[]
	basis.append(np.array([0, 0.5, 0.5, 1]))
	basis.append(np.array([0.5, 0, 0.5, 1]))
	basis.append(np.array([0.5, 0.5, 0, 1]))
	E=np.array([[0, 0.5, 0.5],[0.5,0,0.5],[0.5,0.5,0]])	
	return (E, basis)

def getPrimBasisWZ():
	basis=[]
	basis.append(np.array([0.5, -0.5*np.sqrt(3), 0, 1]))
	basis.append(np.array([0.5, +0.5*np.sqrt(3), 0, 1]))
	basis.append(np.array([0, 0, 1, 1]))
	E=np.array([[0.5, -0.5*np.sqrt(3), 0],[0.5, +0.5*np.sqrt(3), 0],[[0, 0, 1]]])	
	return (E, basis)

#The cluster is built of factor[0]*factor[1]*factor[2] unit cells. The neighborhood basis 'nbBasis' 
#is used to multiply the clusters and form the neighborhood. The neighborhood 'nbhood' is chosen 
#to consist of 125 clusters in order to simulate infinity.
def buildCluster(basis):
	cluster=[]
	for i in range(factor[0]):
		for j in range(factor[1]):
			for k in range(factor[2]):
				x=np.dot(basis[0][0:3],i)
				y=np.dot(basis[1][0:3],j)
				z=np.dot(basis[2][0:3],k)
				r=np.add(np.add(x,y),z)
				r=np.append(r,1)
				cluster.append(r)
	nbBasis=[]
	nbBasis.append(np.dot(basis[0],factor[0]))
	nbBasis.append(np.dot(basis[1],factor[1]))
	nbBasis.append(np.dot(basis[2],factor[2]))
	nbhood=[]
	for i in range(5):
		for j in range(5):
			for k in range(5):
				x=np.dot(nbBasis[0][0:3],i-2)
				y=np.dot(nbBasis[1][0:3],j-2)
				z=np.dot(nbBasis[2][0:3],k-2)
				r=np.add(np.add(x,y),z)
				for l in range(len(cluster)):
					v=np.add(cluster[l][0:3],r)
					v=np.append(v,1)
					nbhood.append(v)
	return (cluster,nbhood)

#We can represent the symmetry operations as permutations of atom locations.
def findPermutations(cluster, nbhood, symOp):
	permList=[]
	for S in symOp:
		newNb=[]
		permutation=[]
		for v in nbhood:
			newNb.append(np.dot(S,v))
		for r in cluster:
			newLoc=[np.array_equal(r,x) for x in newNb].index(True)
			permutation.append(newLoc)
		permList.append([x%N for x in permutation])
	return permList

#This function takes the generator Seitz matrices and determines the symmetry group.
def findAllOp(basicOp,E): 
	Ei=np.linalg.inv(E);
	symOp=basicOp
	for S1 in symOp:
		for S2 in symOp:
			duplicate=0;
			S=np.dot(S1,S2)
			v=np.dot(Ei,S[0:3,3])
			v2=np.array([v[0]%factor[0],v[1]%factor[1], v[2]%factor[2]])
			#v2=np.array([np.sign(v[0])*(v[0]%factor[0]), np.sign(v[1])*(v[1]%factor[1]), np.sign(v[2])*(v[2]%factor[2])])
			#v2=np.array([np.sign(v[0])*(v[0]%1), np.sign(v[1])*(v[1]%1), np.sign(v[2])*(v[2]%1)])
			v3=np.dot(E,v2)
			S[0:3,3]=v3
			for j in range(len(symOp)):
				if np.all(S==symOp[j]) or np.all(-S==symOp[j]):
					duplicate=1
					break
			if duplicate==0:
				symOp.append(S)
	return symOp

def checkInverse(symOp,E):
	Ei=np.linalg.inv(E);
	found=[0]*len(symOp)
	isGroup=1;
	for i in range(len(symOp)):
		Sinv=np.linalg.inv(symOp[i])
		v=np.dot(Ei,Sinv[0:3,3])
		v2=np.array([v[0]%factor[0],v[1]%factor[1], v[2]%factor[2]])
		v3=np.dot(E,v2)
		Sinv[0:3,3]=v3
		for j in range(len(symOp)):
			if np.all(Sinv==symOp[j]) or np.all(-Sinv==symOp[j]):
				found[i]=1;
				break
		if found[i]==0:
			isGroup=0
	if isGroup:
		print('OK')
	else:
		print('ERROR')
	
def findEquivalent(config, permList, symFound):
	equivalent=[]
	for p in permList:
		duplicate=0
		newConfig=[0]*N
		for i in range(N):
			newConfig[i]=config[p[i]]
		for i in range(len(equivalent)):
			if np.all(newConfig==equivalent[i]):
				duplicate=1
				break
		if duplicate==0:
			equivalent.append(newConfig)
			indx=0
			for k in range(N):
				indx+=pow(M,N-k-1)*newConfig[k]
			symFound[indx]=1
	return (equivalent, symFound)

def writePERMLIST(permList):
	print('Writing PERMLIST...                  '),
	file = open('PERMLIST','w')
	for p in permList:
		file.write('[ ')
		for number in p:
			file.write(str(number))
			file.write(' ')
		file.write(']\n')
	file.close()
	print('Done')

def readPERMLIST():
	print('Reading PERMLIST...                  '),
	file = open('PERMLIST','r')
	strPermList=file.read().split(']\n')
	permNumber=len(strPermList)
	permList=[]
	for i in range(permNumber-1):
		strPermList[i]=strPermList[i].split('[')[-1]
		strP=strPermList[i].split(' ')
		p=[]
		for j in range(len(strP)-2):
			p.append(int(strP[j+1]))
		permList.append(p)
	file.close()
	print('Done')
	return permList
	
def main():	
	global factor					
	global N
	global M
	cation  = ['In']
	anion   = ['N','Bi','a']
	struc   = 'zincblende'
	factor  = [2,2,2]					#factoring of N number of interchangeable atoms in the cluster
	N       = 8						#number of interchangeable atoms in the cluster
	M       = max([len(anion),len(cation)])			#number of different types of interchangeable atoms
	(E, basis)       = getPrimBasisZB() 		#input
	checkIsGroup=1
	if(os.path.isfile('PERMLIST')==True):
		print('\nPERMLIST found. Reading permutations.')
		permList         = readPERMLIST()
	if(os.path.isfile('PERMLIST')==False):	
		print('\nPERMLIST not found.')
		basicOp          = getBasicOpZB()			#input
		print('Generating Symmetry Group...         '),
		symOp            = findAllOp(basicOp,E)
		print('Done')
		(cluster,nbhood) = buildCluster(basis)
		print('Finding Atom permutations...         '),
		permList         = findPermutations(cluster, nbhood, symOp) #each symOp appld to the nbhood genrts a prmtatn of clster locations
		print('Done')
		if(checkIsGroup):
			print('Checking group proprieties:')
			print('     Inverse:   '),
			checkInverse(symOp,E)
			print('     Closure:   '),
			symOp2           = findAllOp(symOp,E)
			if(len(symOp2)==len(symOp)):
				print('OK')
			else:
				print('ERROR')
		writePERMLIST(permList)
	print('Finding equivalent clusters...       '),
	symFound         = [0]*pow(M,N)
	factoredConfig   = []
	for i in range(len(symFound)):
		if symFound[i]==0:
			config=np.base_repr(i,base=M)				#obtains the atomic configuration
			complement=[0]*(N-len(config))
			config=complement+[int(y) for y in list(config)]	#fills the remaining config length with zeros
			(equivalent, symFound)=findEquivalent(config, permList,symFound) #equivalent is a list of isoenergetic config
			factoredConfig.append(equivalent)
	print('Done')
	print('\n----------------- RESULTS ----------------\n')
	print('     Representative            Degenerancy\n'),
	total=0
	for i in range(len(factoredConfig)):
		total+=len(factoredConfig[i])
		print(factoredConfig[i][0]),
		print('          '),
		print(len(factoredConfig[i]))
		
	print('\nNumber of total cluster configurations: '),
	print(total)
	print('Clusters with distinct energies: '),
	print(str(len(factoredConfig)) +'\n')
	
	print('Writing POSCAR')
	if(len(cation)>len(anion)):
		atoms1=cation
		atoms2=anion
	else:
		atoms1=anion
		atoms2=cation
	direct='POSCAR_FOLDER'
	if not os.path.exists(direct):
		os.makedirs(direct)
	for i in range(len(factoredConfig)):
		file = open(direct+'/POSCAR#'+str(i),'w')
		representative=factoredConfig[i][0]
		file.write(struc +'\nlattice constant\n')
		for j in range(3):
			file.write(str(basis[j][0])+' '+str(basis[j][1])+' '+str(basis[j][2])+'\n')
		atoms1Pos=[]
		#for j in range(M):
		#	c=[]
		#	locations=[]
		#	for k, l in enumerate(representative):
    		#		if(l==j):
       		#			c.append(k)
		#	for k in c:
		#		locations.append(cluster[k])
		#	atoms1Pos.append(locations)
		#	print(locations)
		#	file.write()
		file.close()
																																																																																	

main()



