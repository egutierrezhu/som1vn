from __future__ import division 
import numpy as np 
import scipy.io.wavfile as wav

from features import mfcc
from numpy import linalg as LA

class TestingNetwork:

	def __init__(self,Nx,Ny,weights,labels):

                
		self.weights = weights
                self.labels = labels

                # lenght of grid X
                self.Nx = Nx    
                # lenght of grid Y
                self.Ny = Ny
                # number neurons X*Y        
                self.N = Nx*Ny
                # deep of neighborhood, p_max=1 for point test
                self.p_max = np.min([Nx,Ny])   

                if self.weights.shape[0] != self.N:
                    raise Warning("The map size must be " + str(N))

	def competitiveProcess(self,xIn):

                d=[]                
                for i in range(self.N):                       
                    d.append(LA.norm(xIn-self.weights[i]))

                r = np.argmin(d)

                lrx = r // self.Ny
                lry = r % self.Ny
                lr = [lrx, lry]

                # topology neighborhood
                labels_path = np.zeros(self.labels.shape[1], dtype=float)
                p = 0                
                while p < self.p_max: 
                    pxy = self.square_path(lrx,lry,p)
                    for q in range(len(pxy)):
                        px = int(pxy[q][0])
                        py = int(pxy[q][1])
                        if (0 <= px) and (px < self.Nx) and (0 <= py) and (py < self.Ny):
                            r_path = int(px*self.Ny+py)
                            labels_path = labels_path+self.labels[r_path]
                    A = np.copy(labels_path)                      
                    labels_aux = self.order_vector(A)
                    if labels_aux[0] > labels_aux[1]:
                        p = self.p_max
                    else:
                        p+=1
 
                return labels_path/np.sum(labels_path)

	def rhombus_path(self,rx,ry,p):
	    
	    if p == 0:
		path = np.zeros((1,2), dtype=float)
		path[0] = [rx, ry]
	    else: 
		path = np.zeros((4*p,2), dtype=float)         
		for k in range(p):
		    path[k] = [rx-p+k,ry+k]
		    path[k+p] = [rx+k,ry+p-k]
		    path[k+2*p] = [rx+p-k,ry-k]
		    path[k+3*p] = [rx-k,ry-p+k]
	    return path

	def square_path(self,rx,ry,p):
	    
	    if p == 0:
		path = np.zeros((1,2), dtype=float)
		path[0] = [rx, ry]
	    else: 
		path = np.zeros((8*p,2), dtype=float)         
		for k in range(2*p):
		    path[k] = [rx-p,ry-p+k]
		    path[k+2*p] = [rx-p+k,ry+p]
		    path[k+4*p] = [rx+p,ry+p-k]
		    path[k+6*p] = [rx+p-k,ry-p]
	    return path

	def hexagon_path(rx,ry,p):
	    
	    if p == 0:
		path = np.zeros((1,2), dtype=float)
		path[0] = [rx, ry]
	    else: 
		path = np.zeros((6*p,2), dtype=float)
		ko = 0         
		for k in range(p):
		    if k == 0:
		        # 2 points
		        path[ko] = [rx-(p-(k // 2)),ry]
		        path[ko+1] = [rx+(p-(k // 2)-(k % 2)),ry]
		        ko+=2
		    else:
		        # 4(p-1) points
		        path[ko] = [rx-(p-(k // 2)),ry-k]
		        path[ko+1] = [rx+(p-(k // 2)-(k % 2)),ry-k]
		        path[ko+2] = [rx-(p-(k // 2)),ry+k]
		        path[ko+3] = [rx+(p-(k // 2)-(k % 2)),ry+k]
		        ko+=4
		# 2(p+1) points
		for l in range(p):
		    path[ko] = [rx-(p-(p // 2))+l,ry-p]
		    path[ko+1] = [rx-(p-(p // 2))+l,ry+p]
		    ko+=2   
		path[ko] = [rx+(p-(p // 2)-(p % 2)),ry-p]
		path[ko+1] = [rx+(p-(p // 2)-(p % 2)),ry+p]         
	    return path

	def order_vector(self,A):
	    for i in range(len(A)):
		for j in range(len(A)):
		    if A[i] > A[j]:
		        aux = A[i]
		        A[i] = A[j]
		        A[j] = aux
	    return A

def testInit(Nx,Ny,filename_som1,filename_lab1):
	#Setup Neural Network
        fsom = open(filename_som1, "rb")
        flab = open(filename_lab1, "rb")
	weights  = np.load(fsom)
        labels  = np.load(flab)
	testNet = TestingNetwork(Nx,Ny,weights,labels)
	return testNet

def extractFeature(soundfile):
	#Get MFCC Feature Array
	(rate,sig) = wav.read(soundfile)
	duration = len(sig)/rate;	
	mfcc_feat = mfcc(sig,rate,winlen=duration/20,winstep=duration/20)
	s = mfcc_feat[:20]
	st = []
	for elem in s:
		st.extend(elem)
	st /= np.max(np.abs(st),axis=0)
	inputArray = np.array([st])
	return inputArray

def feedToNetwork(words,inputArray,testNet):
	#Input MFCC Array to Network
	outputArray = testNet.competitiveProcess(inputArray)

        #print outputArray

	#if the maximum value in the output is less than
	#the threshold the system does not recognize the sound
	#the user spoke

        indexMax = np.argmax(outputArray)
			
	# Mapping each index to their corresponding meaning

        return words[indexMax]

if __name__ == "__main__":

        words = ['backward','forward','go','left','right','stop']


        # Initial values
        # -------------------------------------              
        Nx = 30     #lenght of grid X
        Ny = Nx     #lenght of grid Y  
        print "SOM: " + str(Nx) + "x" + str(Ny)


        filename_som1 = "maps/weights_cmd_6words.npy"
        filename_lab1 = "maps/labels_cmd_6words.npy"
        filename_test = "test_files/test.wav"

        print "Testing: " + filename_test
        
        testNet = testInit(Nx,Ny,filename_som1,filename_lab1)

        inputArray = extractFeature(filename_test)      
        
        testStr = feedToNetwork(words,inputArray,testNet)

        print(testStr)





	


	
		


