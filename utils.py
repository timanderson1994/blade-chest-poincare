import numpy as np
import matplotlib.pyplot as plt

def data_parse(filename, trainSplit = 0.5, returnMatches = False):
    '''
    Data parsing function - usable for all datasets in Chen (2016)
    Input:
        filename - path to data file to read
    
    Output: 
        matches - list of all individual matches in the dataset
        D - dataset "D" from Chen (2016)
    '''
    
    f = open(filename, 'r')
    lines = f.readlines()
    matches_train = []
    matches_test = []
    D_train = []
    D_test = []
    
    matchCount = 0
    
    for i, line in enumerate(lines):
        
        if (line[0:6]=='Player'):
            pass
        elif line[0:3]=='num':
            part1, part2 = line.split()
            if line[0:10] == 'numPlayers':
                numPlayers = int(part2)
            if line[0:8] == 'numGames':
                numGames = int(part2)
        else:
            part1, part2 = line.split()
            player1, player2 = part1.split(':')
            win1, win2 = part2.split(':')
            match_tuple = (int(player1), int(player2), int(win1), int(win2))
            
            if matchCount < numGames*trainSplit:
                matches_train.append(match_tuple)
            else:
                matches_test.append(match_tuple)
            
            matchCount+=1

    # Build training set
    winsA = []
    winsB = []
    matchUps = []
    
    for match in matches_train:
        player1, player2, win1, win2 = match
        
        if (player1, player2) in matchUps:
            ind = matchUps.index((player1, player2))
            winsA[ind] += win1
            winsB[ind] += win2
        else:
            matchUps.append((player1, player2))
            winsA.append(win1)
            winsB.append(win2)
    
    # Build test set
    for i in range(len(matchUps)):
        playerA, playerB = matchUps[i]
        D_train.append((playerA, playerB, winsA[i], winsB[i]))
     
    winsA = []
    winsB = []
    matchUps = []
    
    for match in matches_test:
        player1, player2, win1, win2 = match
        
        if (player1, player2) in matchUps:
            ind = matchUps.index((player1, player2))
            winsA[ind] += win1
            winsB[ind] += win2
        else:
            matchUps.append((player1, player2))
            winsA.append(win1)
            winsB.append(win2)
    
    for i in range(len(matchUps)):
        playerA, playerB = matchUps[i]
        D_test.append((playerA, playerB, winsA[i], winsB[i]))
    
    if returnMatches:
        return numPlayers, numGames, matches_train, D_train, matches_test, D_test
    else:
        return numPlayers, numGames, D_train, D_test
    
def sigmoid(x):
    out = 1 / (1 + np.exp(-x))
    return out

def distance(theta1, theta2, mode = 'Euclidean'):
    '''
    Inputs:
        theta1 - first vector (array like)
        theta2 - second vector (array like)
        mode - either 'Euclidean' or 'Riemannian' depending on which metric
    
    Output:
        dist - distance between theta1 and theta2 under given metric
    '''
    
    if mode == 'Riemannian':
        a = 1 - np.linalg.norm(theta1)**2
        b = 1 - np.linalg.norm(theta2)**2
        dist = np.arccosh(1 + 2 * np.linalg.norm(theta1 - theta2)**2 / a / b)
    else:
        dist = np.linalg.norm(theta1 - theta2)**2
    
    return dist


def ddist(theta1, theta2, mode = 'Euclidean'):
    '''
    Inputs:
        theta1 - first vector (array like)
        theta2 - second vector (array like)
        mode - either 'Euclidean' or 'Riemannian' depending on which metric
        
    Outputs:
        ddist1 - gradient of distance w.r.t. first input vector
        ddist2 - gradient of distance w.r.t. second input vector
        
    '''
    
    if mode == 'Riemannian':
        # constants used in computing gradient
        alpha = 1 - np.linalg.norm(theta1)**2
        beta = 1 - np.linalg.norm(theta2)**2
        gamma = 1 + 2 / alpha / beta * np.linalg.norm(theta1 - theta2)**2

        # compute gradients
        ddist1 = 4/beta/np.sqrt(gamma**2-1)*((np.linalg.norm(theta2)**2 - 2*theta1.dot(theta2)
                                              + 1)/alpha**2 * theta1 - 1 / alpha*theta2)
        ddist2 = 4/alpha/np.sqrt(gamma**2-1)*((np.linalg.norm(theta1)**2 - 2*theta1.dot(theta2)
                                              + 1)/beta**2 * theta2 - 1 / beta*theta1)
    else:
        ddist1 = 2 * (theta1 - theta2)
        ddist2 = -2 * (theta1 - theta2)
        
    return ddist1, ddist2

    
def matchup(playerA, playerB, mode = 'Euclidean'):
    '''
    Inputs:
        playerA - tuple containing blade, chest, and strength parameter for playerA
        playerB - tuple containing blade, chest, and strength parameter for playerB
        mode - either 'Euclidean' (default) or 'Riemannian' for type of embedding
    
    Output:
        score - value of matchup function
    
    '''
    
    bladeA, chestA, gammaA = playerA
    bladeB, chestB, gammaB = playerB

    score = distance(bladeB, chestA, mode = mode) - distance(bladeA, chestB, mode = mode) + gammaA - gammaB
    
    return score

def dmatchup(playerA, playerB, mode = 'Euclidean'):
    '''
    Inputs:
        playerA - tuple containing blade, chest, and strength parameter for playerA
        playerB - tuple containing blade, chest, and strength parameter for playerB
        mode - either 'Euclidean' or 'Riemannian' for type of embedding
    
    Outputs:
        playerAout - tuple containing derivatives for blade, chest, and strength parameter for A
        playerBout - tuple containing derivatives for blade, chest, and strength parameter for B
    '''
    
    bladeA, chestA, gammaA = playerA
    bladeB, chestB, gammaB = playerB
    
    dchestA, dbladeB  = ddist(chestA, bladeB, mode = mode)
    dbladeA, dchestB = ddist(bladeA, chestB, mode = mode)
    
    dbladeA *= -1
    dchestB *= -1
    
    dgammaA = 1
    dgammaB = -1
    
    playerAout = (dbladeA, dchestA, dgammaA)
    playerBout = (dbladeB, dchestB, dgammaB)
    
    return playerAout, playerBout

def loss(playerA, playerB, wins, mode = 'Euclidean'):
    '''
    Inputs:
        playerA - list of tuples containing parameters for the batch or minibatch for player A
        playerB - list of tuples containing parameters for the batch or minibatch for player A
        wins - list of tuples containing na and nb values
        
    Output:
        lossVal - value of the loss
    
    Note: function is overloaded to accommodate both batch updates/evaluations (batch GD) 
        or single evaluations (for SGD)
    '''
    if isinstance(playerA, list) and isinstance(playerB, list):
        lossVal = 0
        for i in range(len(playerA)):
            match = matchup(playerA[i], playerB[i], mode = mode)
            na, nb = wins[i]
            lossVal += na*np.log(1 + np.exp(-match)) + nb*np.log(1 + np.exp(match))
    else:
        match = matchup(playerA, playerB, mode = mode)
        na, nb = wins
        lossVal = na*np.log(1 + np.exp(-match)) + nb*np.log(1 + np.exp(match))
    
    return lossVal


def dloss(playerA, playerB, wins, mode = 'Euclidean'):
    '''
    Inputs:
        playerA - list of tuples containing parameters for the batch or minibatch for player A
        playerB - list of tuples containing parameters for the batch or minibatch for player A
        wins - list of tuples containing na and nb values
        
    Output:
        dplayerA - gradients for player A's parameters (either list of tuples or a tuple)
        dplayerB - gradients for player B's parameters (either list of tuples or a tuple)
        
    '''
    
    if isinstance(playerA, list) and isinstance(playerB, list):
        # Using recursion in the loop cleans up implementation
        for i in range(len(playerA)):
            dplayerA[i], dplayerB[i] = dloss(playerA[i], playerB[i], wins[i], mode = mode)
    
    else:
        match = matchup(playerA, playerB, mode = mode)
        na, nb = wins
        dl = -na * np.exp(-match) / (1 + np.exp(-match)) + nb * 1 / (1 + np.exp(-match))
        
        dmatchA, dmatchB = dmatchup(playerA, playerB, mode = mode)
        
        dbladeA, dchestA, dgammaA = dmatchA
        dbladeB, dchestB, dgammaB = dmatchB
        
        dbladeA *= dl
        dchestA *= dl
        dgammaA *= dl
        
        dbladeB *= dl
        dchestB *= dl
        dgammaB *= dl
        
        dplayerA = (dbladeA, dchestA, dgammaA)
        dplayerB = (dbladeB, dchestB, dgammaB)
    
    return dplayerA, dplayerB


def proj(theta, epsilon = 1e-3):
    '''
    Inputs:
        theta - vector to scale
        epsilon - constant for numerical stability
        
    Output:
        thetaOut - scalaed theta
    
    '''
    
    if np.linalg.norm(theta) >= 1:
        thetaOut = theta / np.linalg.norm(theta) - epsilon
    else:
        thetaOut = theta
    
    return thetaOut


def SGD_update(theta, dtheta, alpha, mode = 'Euclidean'):
    '''
    Inputs: 
        theta - parameter at iteration t
        dtheta - gradient at iteration t
        alpha - learning rate at iteration t
        mode - 'Euclidean or 'Riemannian'
    
    Outputs:
        thetaNew - updated theta
    
    '''
    
    if mode == 'Riemannian':
        thetaNew = proj(theta - alpha*(1 - np.linalg.norm(theta)**2)**2 / 4 * dtheta)
    else:
        thetaNew = theta - alpha*dtheta
        
    return thetaNew



def evaluate(y, yhat):
    '''
    Function to evaluate accuracy of predictions
    Inputs:
        y - true labels
        yhat - predicted labels
        
    Outputs:
        acc - accuracy
    '''
    
    eqs = np.equal(y,yhat)
    acc = np.mean(eqs.astype(float))
    return acc


class blade_chest:
    '''
    Class to hold and update blade-chest embedding parameters (blade/chest vectors and gamma's)
    
    '''
    
    def __init__(self, numberOfPlayers, dim, bias = True, initParam = 1e-2, BT = False):
        '''
        Initialize the blade_chest class
        Inputs:
            numberOfPlayers - total number of players for which to initialize parameters
            dim - dimensionality of the embeddings
        '''
        self.BT = BT
        if self.BT:
            self.blades = np.zeros((numberOfPlayers,dim))
            self.chests = np.zeros((numberOfPlayers,dim))
            self.gammas = initParam*(np.random.uniform(size = (numberOfPlayers)) - 0.5)
            self.bias = bias
        else:
            self.blades = initParam*(np.random.uniform(size = (numberOfPlayers,dim)) - 0.5)
            self.chests = initParam*(np.random.uniform(size = (numberOfPlayers,dim)) - 0.5)
            self.gammas = np.zeros(numberOfPlayers)
            self.bias = True
        
    
    def SGD_optimizer(self, playerAnum, playerBnum, wins, alpha, mode = 'Euclidean', reg = 0):
        '''
        Inputs:
            playerA - ID number of the first player
            playerB - ID number of the second player
            wins - tuple or list of tuples containing na and nb values
            mode - 'Euclidean' or 'Riemannian' for type of embedding

        Outputs

        '''
        
        # Compute derivatives
        playerA = (self.blades[playerAnum,:], self.chests[playerAnum,:], self.gammas[playerAnum])
        playerB = (self.blades[playerBnum,:], self.chests[playerBnum,:], self.gammas[playerBnum])
        
        dplayerA, dplayerB = dloss(playerA, playerB, wins, mode = mode)
        
        # PLAYER A
        dblade, dchest, dgamma = dplayerA
        dRegBlade, dRegChest = ddist(self.blades[playerAnum,:], self.chests[playerAnum,:], mode = mode)
        
        if not self.BT:
            self.blades[playerAnum,:] = SGD_update(self.blades[playerAnum,:], dblade+reg*dRegBlade, alpha, mode = mode)
            self.chests[playerAnum,:] = SGD_update(self.chests[playerAnum,:], dchest+reg*dRegChest, alpha, mode = mode)
        
        if self.bias:
            self.gammas[playerAnum] = SGD_update(self.gammas[playerAnum], dgamma, alpha)
        
        
        # PLAYER B
        dblade, dchest, dgamma = dplayerB
        dRegBlade, dRegChest = ddist(self.blades[playerBnum,:], self.chests[playerBnum,:], mode = mode)
        
        if not self.BT:
            self.blades[playerBnum,:] = SGD_update(self.blades[playerBnum,:], dblade+reg*dRegBlade, alpha, mode = mode)
            self.chests[playerBnum,:] = SGD_update(self.chests[playerBnum,:], dchest+reg*dRegChest, alpha, mode = mode)
        
        if self.bias:
            self.gammas[playerBnum] = SGD_update(self.gammas[playerBnum], dgamma, alpha)
        
    def accuracy(self, D, mode = 'Euclidean'):
        '''
        Inputs:
            D - data over which to evaluate the accuracy. List of tuples (playerA, playerB, na, nb)
            
        Outputs:
            acc - accuracy 
            
        '''
        
        acc = 0
        Nprime = 0
        
        for i in range(len(D)):
            playerAnum, playerBnum, na, nb = D[i]
            
            playerA = (self.blades[playerAnum,:], self.chests[playerAnum,:], self.gammas[playerAnum])
            playerB = (self.blades[playerBnum,:], self.chests[playerBnum,:], self.gammas[playerBnum])
            prob = sigmoid(matchup(playerA, playerB, mode = mode))
            acc += na*int(prob > 0.5) + nb*int(prob <= 0.5)
            Nprime += na + nb
        
        return acc/Nprime
    
    
    def naive_train(self, numPlayers, Dtrain):
        '''
        Input:
            D - datapoints for predictions
        '''
        
        numer = np.ones((numPlayers, numPlayers))
        denom = 2*np.ones((numPlayers, numPlayers))

        for d in Dtrain:
            playerAnum, playerBnum, na, nb = d
            numer[playerAnum, playerBnum] += na
            denom[playerAnum, playerBnum] += na + nb
            numer[playerBnum, playerAnum] += nb
            denom[playerBnum, playerAnum] += na + nb

        self.naiveMtx = numer / denom
    
    def naive_eval(self, Deval):
        '''
        Input:
            D - datapoints for predictions
        '''
        matchMtx = self.naiveMtx
        acc = 0
        Nprime = 0

        for d in Deval:
            playerAnum, playerBnum, na, nb = d
            acc += na*int(matchMtx[playerAnum, playerBnum] > 0.5) + nb*int(matchMtx[playerAnum, playerBnum] <= 0.5)
            Nprime += na + nb

        return acc/Nprime
    
    
