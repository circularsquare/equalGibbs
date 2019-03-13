''''
attempt to fit equal gibbs data with gradient descent
andrew chen
gerald van hecke
started 2/17/19
'''


'''
todo:
make cost function encourage simple solutions... maybe weight <100 delG points much more or use more guess points
discourage huge delgs far from points
have an integrated way of guessing instead of desmos???
speeeeeed
'''

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import random
import math
import time
import sys


#points to be fitted, in [x, y] form
#x is mole fraction (btw 0 and 1), y is temperature (200-800ish)
'''data1 = [[0, 400],[.1, 380],[.1, 317],[.1, 293],
        [.2, 373],[.2, 325],[.2, 260],[.3, 225],[.3, 325],[.3, 365],
        [.4, 328],[.5, 330],[.6, 331],[.7, 332],[.8, 333],[.9, 333],[1, 333],
        [.4, 362],[.5, 360],[.6, 356],[.7, 354],[.8, 352],[.9, 351], [1, 350]]
data2 = [[0, 400], [.05, 402], [.1, 405], [0, 430], [.05, 422], [.15, 410], [.15, 417], [.02, 427],
        [1, 350], [.95, 354], [.9, 361], [.85, 369], [.85, 375], [.9, 382], [.95, 383], [1, 384]]'''
#goal2: 400, 350, 50, 50, -500, -300, -2, -2, 1, 1
#best so far is 362, 397, 47, 46, .57, .29, -7, -2, 0s

#the one called "data" actually runs... make it read from files or something
data = [[0, 539], [.1, 533], [.2, 523], [.3, 517], [.42, 509], [.52, 504], [.64, 497], [.75, 487], [.84, 483], [.95, 473]]

runningParams = [[0]*13]*10 #where all the seeds are stored, can change number of seeds
params = [0]*13 #set of params selected as best out of runningParams... again get rid of global
#params are in order of t1, t2, s1, s2, a1, a2, b1, b2, c1, c2, A1, A2, A3 (can disable last n constants by commenting in init and editing adjust function)
sensitivities = [10, 10, 1, 1, .01, .01, .02, .02, 500, 500, .1, .1, .1] #approximate scales that params vary on
avgDelG = 10000000 #measure of avg delG over the region of interst

calcCounter = 0
#so that you only calculate the avgDelG once in a while instead of every cost call
#def find better way to do this

def init(paramsa): #randomly places parameters... ranges should be tweaked
    paramsa[0] = 200 + 600*random.random()   #t1
    paramsa[1] = 200 + 600*random.random()   #t2
    paramsa[2] = -50 + 100*random.random()   #s1
    paramsa[3] = -50 + 100*random.random()   #s2
    paramsa[4] = -1 + 2*random.random()      #a1, reentrance
    paramsa[5] = -1 + 2*random.random()      #a2
    paramsa[6] = -10 + 20*random.random()    #b1
    paramsa[7] = -10 + 20*random.random()    #b2
    #paramsa[8] = -200 + 400*random.random()  #c1
    #paramsa[9] = -200 + 400*random.random()  #c2
    #paramsa[10] = -10 + 20*random.random()  #A1, needed for reentrant bulges that tilt up or down
    #paramsa[11] = -5 + 10*random.random()   #A2
    #paramsa[12] = -2 + 4*random.random()    #A3
    calcAvgDelG(paramsa)

def delG(x, y, paramsa = None): #calculates difference in gibbs energy is for given x, y
    if paramsa == None:
        paramsa = params
    if(paramsa[0]<=0):
        print("params r 0s")
        print(paramsa)
        sys.exit()
    dt1 = y-paramsa[0]
    dt2 = y-paramsa[1]
    log1 = math.log(y/paramsa[0])
    g1 = (x) * (-paramsa[2]*(dt1) + paramsa[4]*(dt1-y*log1) - (paramsa[6]/2)*dt1*dt1 - (paramsa[8]/2)*(dt1*dt1)/(y*paramsa[0]*paramsa[0]))
    log2 = math.log(y/paramsa[1])
    g2 = (1-x)*(-paramsa[3]*(dt2) + paramsa[5]*(dt2-y*log2) - (paramsa[7]/2)*dt2*dt2 - (paramsa[9]/2)*(dt2*dt2)/(y*paramsa[1]*paramsa[1]))
    rpart = paramsa[10]*x*(1-x) + paramsa[11]*x*(1-x)*(1-2*x) + paramsa[12]*x*(1-x)*(1-2*x)*(1-2*x)
    return  g1+g2+rpart

def calcAvgDelG(paramsA): #CURRENTLY USES LOG of delG! u can disable log they both seem to work ok
    global calcCounter
    calcCounter += 1
    if calcCounter>=200:    #so this function only really runs every once in a while
        numRands = 1000     #number of random points to estimate average cost
        outCost = 0
        for i in range(numRands):
            outCost += math.log(abs(delG(random.random(), 100+900*random.random(), paramsA)))
        avgDelG = (outCost/numRands)
        calcCounter = 0

#divides sum of delGs of the data by the average delG in the region x 0-1 and y 100-1000
def costO(points, paramsa):#delG cost
    dataDelG = 0
    calcAvgDelG(paramsa)
    for point in points:
        pointG = delG(point[0], point[1], paramsa)
        dataDelG += pointG*pointG
        if(pointG>math.pow(10, 30)):
            print('ripperino')
            print(point)
            print(paramsa)
            sys.exit()
    return (dataDelG/len(points))/math.pow(avgDelG, .4)

def costUnused(points, paramsa):#orthogonal cost (doesn't rn work)
    tDist = 0
    xDist = 0
    sumDists = 0
    tolerance = 100
    for p in points:
        xDist=1
        tDist=1000
        tStart = p[1]
        xStart = p[0]
        for j in range(0, 1000, 4):
            if delG(xStart, tStart+j, paramsa)<tolerance:
                tDist=j
                break
            if(tStart>j):
                if delG(xStart, tStart-j, paramsa)<tolerance:
                    tDist=j
                    break
        i=0
        while (i < 1):
            if delG(xStart-i, tStart, paramsa)<tolerance:
                xDist=i
                break
            if delG(xStart+i, tStart, paramsa)<tolerance:
                xDist=i
                break
            i+=.01
        if ((xDist==0)|(tDist==0)):
            sumDists += 0
        elif (xDist*400<tDist):
            sumDists += xDist*400
        else:
            sumDists += tDist
    return sumDists

def adjust(paramsa):
    gradient=[0]*13
    calcAvgDelG(paramsa)
    cost0 = costO(data, paramsa)
    for i in range(8): #change the number it iterates through to leave out parameters at the end of the array
        diff = sensitivities[i]*.0001
        paramsa[i] += diff
        gradient[i] = (costO(data, paramsa)-cost0)/(sensitivities[i]*.00002)
        paramsa[i] -= diff
    for i in range(13):
        movement = sensitivities[i]*gradient[i]*.0000002 #lower this if it blows up, increase it if it runs fine
        if movement>100:
            movement=100
        if movement<-100:
            movement=-100
        paramsa[i] -= movement
    if(paramsa[0]<=0): #if this happens ull blow up prob
        paramsa[0]=1
    if(paramsa[1]<=0):
        paramsa[1]=1

def tabulate(x, y, f):
    return np.vectorize(f)(*np.meshgrid(x, y))

#plots log of delG (and -log(-delG) for negatives, and delGs between -1 and 1 show as 0)
def graph2():
    plt.figure()
    xlist = np.linspace(0, 1, 120)
    ylist = np.linspace(100, 800, 200)
    zlist = tabulate(xlist, ylist, delG)
    for i in range(len(zlist)):
        for j in range(len(zlist[0])):
            z = zlist[i][j]
            if(z<-1):
                zlist[i][j]=-math.log(-z)
            elif(z<0):
                zlist[i][j]=0
            elif(z<1):
                zlist[i][j]=0
            else:
                zlist[i][j]=math.log(z)
    plt.contourf(xlist, ylist, zlist,cmap=cm.RdBu_r, vmin=-10, vmax=10)
    plt.colorbar()
    dataXs = [0]*len(data)
    dataYs = [0]*len(data)
    for i in range(len(data)):
        dataXs[i] = data[i][0]
        dataYs[i] = data[i][1]
    plt.plot(dataXs, dataYs, 'ro')
    plt.show()

#plots the dG as the z coord
def graph3(paramsa):
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plotXs = [0]*2121
    plotYs = [0]*2121
    plotZs = [0]*2121
    for i in range(101):
        for j in range(21):
            plotXs[i*21+j] = .05*j
            plotYs[i*21+j] = 100 + 9*i
            plotZs[i*21+j] = delG(.05*j, 100+9*i, paramsa)
    ax.scatter(plotXs, plotYs, plotZs, zdir='z', s=20, c=None, depthshade=True)
    plt.show()

def dCost(params):
    prevCost = costO(data, params)
    adjust(params)
    newCost = costO(data, params)
    return newCost/prevCost

def runParams():
    runAllSeeds = 1000 #number of times to adjust seeds before pruning
    tolerance = .999995 #if adjust helps lower cost by less than this number, stop adjusting
    #increase to make it run for longer

    startTime0 = time.time()
    minCost = math.pow(10, 40)
    minAttempt = -1
    minParams = [0]*13
    global calcCounter #find a better way to do this
    for i in range(len(runningParams)):
        calcCounter=10000 #this is to make calcAvgDelG run
        init(runningParams[i])
        for j in range(runAllSeeds):
            adjust(runningParams[i])
        print(costO(data, runningParams[i]))
        if costO(data, runningParams[i])<minCost:
            minCost=costO(data, runningParams[i])
            minAttempt=i
            minParams = [runningParams[i][j] for j in range(13)]
    global params
    params = minParams
    for j in range(1000):
        adjust(params)
    print(costO(data, params))
    while(dCost(params)<tolerance):
        for j in range(500):
            adjust(params)
        print(costO(data, params))
    endTime0 = time.time()
    print("--- %s seconds total ---" % (endTime0-startTime0))
    print(params)
    graph2()
    return params



runParams()
