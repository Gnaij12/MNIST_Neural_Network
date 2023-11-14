import ast
from cmath import pi
import sys
from matplotlib import pyplot as plt
import numpy as np
from math import exp

def pretty_print_tt(table): #Dictionary
    for i in range(len(list(table.keys())[0])):
        print("ln"+str(i),end=" ")
    print("  out")
    for tup, val in table.items():
        for i in tup:
            print(i,end="   ")
        print("| " + str(val))

def truth_table(bits,n):
    binaryn = format(n,"b") #in str form
    if len(binaryn) < 2**bits:
        binaryn = "0" * (2**bits-len(binaryn)) + binaryn
    tt = {}
    for i in range(2**bits-1,-1,-1):
        bn = tuple(map(int,tuple(format(i,"b"))))
        if len(bn) < bits:
            bn = (0,) * (bits-len(bn)) + bn
        tt[bn] = int(binaryn[2**bits-i-1])
    return tt
    
def step(num):
    if num > 0:
        return 1
    return 0

def sigmoid(num):
    return 1/(1+exp(-1*num))

def sigmoidprime(num):
    return sigmoid(num) * (1-sigmoid(num))

def perceptron(A, w, b, x):
    total = 0
    for i in range(len(w)):
        total+=w[i] * x[i]
    return A(total+b)

def pnet(A,wlist,blist,x):
    vA = np.vectorize(A)
    a = [x]
    for i in range(1,len(wlist)):
        a.append(vA(a[i-1]@wlist[i]+ blist[i]))
    return a[len(a)-1]

def pnetAlist(A,wlist,blist,x):
    vA = np.vectorize(A)
    alist = [x]
    for i in range(1,len(wlist)):
        temp = alist[i-1]@wlist[i]
        alist.append(vA(temp+ blist[i]))
    return alist

def check(n, w, b):
    tt = truth_table(len(w),n)
    correct = 0
    for x in tt.keys():
        if perceptron(step,w,b,x) == tt[x]:
            correct+=1
    return correct/len(tt)

def checkwtt(tt,w,b):
    correct = 0
    for x in tt.keys():
        if perceptron(step,w,b,x) == tt[x]:
            correct+=1
    return correct/len(tt)

def trainperceptron(tt,w,b):
    prevw = [-1]
    prevb = -1
    learnrate = 1
    w = list(w)
    total = 0
    while (prevw != w or prevb != b) and total < 100:
        prevw = w.copy()
        prevb = b
        for x,result in tt.items():
            f = perceptron(step,w,b,x)
            if f != result:
                diff = result - f
                for i in range(len(w)):
                    w[i]=w[i] + diff * learnrate * x[i]
                b += diff * learnrate
        total+=1
    return tuple(w),b

def makegraph(i,tt,w,b):
    plt.xlim(-2.1,2)
    plt.ylim(-2.1,2)
    plt.title("#"+str(i))
    for i in range(-20,20):
        for j in range(-20,20):
            x = i/10
            y = j/10
            if (x == 1 and (y == 0 or y == 1)) or (x == 0 and (y == 0 or y == 1)):
                if tt[(int(x),int(y))] == 1:
                   plt.scatter(x,y,c="green",s=20)
                else:
                    plt.scatter(x,y,c="red",s=20)
            else:
                if perceptron(step,w,b,(x,y)) == 1:
                    plt.scatter(x,y,c="green",s=5)
                else:
                    plt.scatter(x,y,c="red",s=5)
    plt.show()

def runallperceptrons(bits):
    tts = []
    for n in range(2**(2**bits)):
        tts.append(truth_table(bits,n))
    correct = 0
    for i,tt in enumerate(tts):
        w,b = trainperceptron(tt,(0,)*bits,0)
        makegraph(i,tt,w,b)
        if checkwtt(tt,w,b) == 1:
            correct+=1
    # plt.show()
    # print("%s Possible functions; %s can be correctly modeled." %(2**(2**bits),correct))

def makefunct(v,gradvals):
    def funct(learnrate):
        return v - learnrate * np.array([[gradvals[0],gradvals[1]]])
    return funct

def minimize(funct,grad,v):
    learnrate = 0.1
    xgrad,ygrad = grad[0][0](v),grad[0][1](v)
    print("At: %s    Gradient: %s" %((v[0][0],v[0][1]),(xgrad,ygrad)))
    while (xgrad**2+ygrad**2)**0.5 > 10**-8:
        f = makefunct(v,(xgrad,ygrad))
        learnrate,temp = one_d_minimize(f,0,1,10**-8,grad)  #Line optimzation here
        v = v - learnrate * np.array([[xgrad,ygrad]])
        xgrad,ygrad = grad[0][0](v),grad[0][1](v)
        print("At: %s    Gradient: %s" %((v[0][0],v[0][1]),(xgrad,ygrad)))

def one_d_minimize(f,left,right,tolerance,grad): #Actual line optimization
    diff = right - left
    if diff < tolerance:
        return (right+left)/2,(f(right)+f(left))/2 
    leftval = f(left + 1/3 * diff)
    rightval = f(left + 2/3 * diff)
    leftxgrad,leftygrad = grad[0][0](leftval),grad[0][1](leftval)
    rightxgrad,rightygrad = grad[0][0](rightval),grad[0][1](rightval)
    if leftxgrad**2+leftygrad**2 > rightxgrad**2+rightygrad**2:
        return one_d_minimize(f,left+1/3 * diff,right,tolerance,grad)
    return one_d_minimize(f,left,left + 2/3 * diff,tolerance,grad)

def roundnum(num):
    if num >= 0.5:
        return 1
    return 0

def findmisclassified(out,y):
    misclassified = 0
    for i in range(len(y)):
        if np.amax(y[i]) != np.amax(out[i]):
            misclassified+=1
    return misclassified

def train(A,Aprime,wlist,blist,x,y,epochs):
    f = open("MNISTwbs.txt","a")
    latestf = open("MNISTwbslatest.txt","w")
    for i in range(epochs):
        print(i)
        for j in range(len(x)):
            alist,wlist,blist = forwardPropagate(A,Aprime,wlist,blist,x[j],y[j],0.1)
        putintofile(f,wlist)
        putintofile(f,blist)
        putintofile(latestf,wlist)
        putintofile(latestf,blist)
        # print(alist[len(alist)-1]) 
        # print(error)
    alist = pnetAlist(A,wlist,blist,x)
    print(np.vectorize(roundnum)(alist[len(alist)-1]))

def trainC(A,Aprime,wlist,blist,x,y,epochs):
    vRn = np.vectorize(roundnum)
    learnrate = 0.01
    # f = open("circlebest.txt","w")
    for i in range(epochs):
        alist,wlist,blist,error = forwardPropagate(A,Aprime,wlist,blist,x,y,learnrate if i == 0 else learnrate/(((i-1)*10/epochs)+1))
        print(alist[len(alist)-1])
        # f.write(str(alist[len(alist)-1]) + "\n")
        print(error)
        alist = pnetAlist(A,wlist,blist,x)
        misclassified = 0
        out = vRn(alist[len(alist)-1])
        for i in range(len(y)):
            if y[i][0] != out[i][0]:
                misclassified+=1
        print("Misclassified points: %s" %misclassified)
        # f.write("Misclassified points: "+ str(misclassified) + "\n")
    # f.write(wlist)

def forwardPropagate(A,Aprime,wlist,blist,x,y,learnrate):
    alist = pnetAlist(A,wlist,blist,x)
    output = alist[len(alist)-1]
    # error = (np.linalg.norm(y-output) **2) / 2
    # print(error)
    blist,wlist = backPropagate(A,Aprime,alist,wlist,blist,learnrate,y)
    return alist,wlist,blist

def backPropagate(A,Aprime,alist,wlist,blist,learnrate,y):
    vAprime = np.vectorize(Aprime)
    dot = alist[len(alist)-2]@wlist[len(alist)-1]+blist[len(alist)-1]
    deltalist = [None] * (len(alist)-1) + [vAprime(dot) * (y-alist[len(alist)-1])]
    for i in range(len(alist)-2,0,-1):
        dot = alist[i-1]@wlist[i] + blist[i]
        deltalist[i] = vAprime(dot) * (deltalist[i+1]@wlist[i+1].transpose())
        print(dot)
    for i in range(1,len(deltalist)):
        delta = deltalist[i]
        blist[i] = blist[i] + learnrate * delta
        temp = learnrate * alist[i-1].transpose() @ delta
        wlist[i] = wlist[i] + temp
    return blist,wlist

def makenetwork(layers):
    wlist = [None]
    blist = [None]
    for i in range(len(layers)):
        if i < len(layers)-1:
            wlist.append(2 * np.random.rand(layers[i], layers[i+1]) - 1)
        if i > 0:
            blist.append(2 * np.random.rand(1, layers[i]) - 1)
    return wlist,blist

def takefromfile(filename):
    y = []
    x = []
    with open(filename) as f:
        for line in f:
            vals = ast.literal_eval(line)
            out = (0,)*vals[0]+(1,) + (0,)*(9-vals[0])
            y.append(np.asarray(out))
            vals = vals[1:]
            x.append(np.asarray((vals,))/255)
    return x,y

def takewblistfromfile(filename):
    wlist = []
    blist = []
    with open(filename) as f:
        listf = list(f)
        lenf = len(listf)
        for i,line in enumerate(listf):
            if i > lenf -3:
                line = line.split("-[[")
                templist = [None]
                for j,v in enumerate(line):
                    v = "[[" + v
                    if j > 0:
                        templist.append(np.asarray(ast.literal_eval(v)))
                if i % 2 == 0: #Wlist!
                    wlist = templist
                else:
                    blist = templist
    return wlist,blist

def putintofile(f,putin):
    for i,v in enumerate(putin):
        if i == 0:
            f.write(str(v) + "/")
        elif i == len(putin)-1:
            f.write(str(v.tolist()))
        else:
            f.write(str(v.tolist()) + "/")
    f.write("\n")

def testperceptron(A,wlist,blist,x,y):
    alist = pnetAlist(A,wlist,blist,x)
    out = alist[len(alist)-1]
    mis = findmisclassified(out,y)
    return mis,len(y)

wlist,blist = makenetwork([784,300,100,10])
# wlist,blist = takewblistfromfile("MNISTwbs.txt")
# wlist,blist = makenetwork([5,2,1])
# f = open("test.txt","w")
# putintofile(f,wlist)
# putintofile(f,blist)
# print("yo")
# print(wlist)
# print(wlist[2])
# print(blist)
x,y = takefromfile("mnist_train.csv")
train(sigmoid,sigmoidprime,wlist,blist,x,y,1000)
