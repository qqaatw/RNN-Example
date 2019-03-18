import math
import numpy as np

Input = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]  # Input Vectors.
'''
U = [[0.287027, 0.846060, 0.572392, 0.486813],  # Output Weights
     [0.902874, 0.871522, 0.691079, 0.189980],
     [0.537524, 0.092240, 0.558159, 0.491528]]

W = [[0.427043, 0.352067, 0.435679],  # Hidden Weights
     [0.213856, 0.857639, 0.365739],
     [0.387357, 0.938756, 0.457934]]  

V = [[0.371680, 0.974829, 0.830035],  # Input Weights
     [0.391410, 0.282586, 0.659836],
     [0.649850, 0.098216, 0.334287],
     [0.912660, 0.325816, 0.144630]]

U = np.array(U)
W = np.array(W)
V = np.array(V)
'''

B = [0.567001]  # Bais


U = np.random.random_sample((3, 4))
W = np.random.random_sample((3, 3))
V = np.random.random_sample((4, 3))


St = [np.zeros((3, 1)) for i in range(len(Input))]
Ot = [np.zeros((3, 1)) for i in range(len(Input))]
Et = [np.zeros((3, 1)) for i in range(len(Input))]

def dlogit(output):  # dlogit (tanh)
     return 1-(output ** 2)

def dh(x, t):  # dh
     return x * dlogit(St[t]) 

def ErrorO(ot, otp):  # derivative of cross-entropy with softmax
     return ot - otp

def crossentropy(ot, otp):
     return -np.mean(np.sum(ot * np.log(otp)))

def categorical_crossentropy(ot, otp):
     output = [];
     for i in range(len(ot)):
          output.append(ot[i] * np.log(otp[i][0]))
     return output
     
def softmax(otp):
     return np.exp(otp)/sum(np.exp(otp))

def fit(Input, Label, Batch, alpha):
     global St, Ot, Et, U, W, V
     initialization = True
     for bc in range(Batch):
          OverallError = 0
          for t in range(len(Input)):
               if t == 0 and initialization == True: # if initialization, St=0
                    initialization = False
                    hiddenValue = np.matmul(W, np.zeros((3, 1)))
               else:
                    hiddenValue = np.matmul(W, np.array(St[t-1]))

               inValue = np.matmul(U, np.array(Input[t])[np.newaxis].T)
               baisValue = B

               StateValue = hiddenValue + inValue + baisValue
               StateValue = np.tanh(StateValue)
               St[t] = StateValue

               outValue = softmax(np.matmul(V, StateValue))
               Ot[t] = outValue

               Et[t] = categorical_crossentropy(Label, Ot[t])
               OverallError += np.abs(sum(Et[t]))
          
          # BPTT
          # http://ir.hit.edu.cn/~jguo/docs/notes/bptt.pdf
          dEdV = np.zeros((V.shape))
          dEdW = np.zeros((W.shape))
          dEdU = np.zeros((U.shape))

          for t in range(len(Input)-1, -1, -1):
               # Calculate output layer's derivative.
               # Derivative of cross entropy loss with softmax has
               # a very simple and elegant expression f'(t)=ot-otp.
               # https://deepnotes.io/softmax-crossentropy
               dEdV += np.outer(ErrorO(Label[np.newaxis].T, Ot[t]), St[t].T)
               
               # Calculate hidden/input layer's derivative.
               delta_s = dh(np.matmul(V.T, ErrorO(Label[np.newaxis].T, Ot[t])), t) # V.T(3X4) X Eo(4X1) * St'(3X1) = 3X1
               
               # Because of W/U relate to previous time-step,
               # we use a iteration to calculate it. 
               for bptt_t in range(t, -1, -1):
                    #if bptt_t-1 >= 0:
                    dEdW += np.outer(delta_s, St[bptt_t-1].T) # delta_s(3X1) X St[bptt_t-1].T(1X3) = 3X3 Matrix
                    dEdU += np.outer(delta_s, Input[bptt_t])
                    # Update derivative.
                    delta_s = dh(np.matmul(W.T, delta_s), bptt_t-1)
          
          # Update Weights
          U = U + alpha * dEdU
          V = V + alpha * dEdV
          W = W + alpha * dEdW
          print('Epoch=', bc, 'Loss:', OverallError)
     print('O[-1]:\n', Ot[-1])
          
          
if __name__ == "__main__":
     fit(Input, np.array([0,1,0,0]), 10, 1)


