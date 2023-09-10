import numpy as np


# ==========================
# CART Tree
class DecisionTree:
    def __init__(self, minLeafSize=5, maxDepth=10):
        self.nodes = []
        self.root = 0
        self.depth = 0
        self.minLeafSize = minLeafSize
        self.maxDepth = maxDepth

    def fit(self, X, y):
        self.nodes.append(Node(X, y))
        self.growTree(self.root)

    def addNodeLeft(self, parentNodeIdx, node):
        newNodeIdx = len(self.nodes)
        self.nodes[parentNodeIdx].insertLeft(newNodeIdx)
        self.nodes.append(node)

        return newNodeIdx

    def addNodeRight(self, parentNodeIdx, node):
        newNodeIdx = len(self.nodes)
        self.nodes[parentNodeIdx].insertRight(newNodeIdx)
        self.nodes.append(node)

        return newNodeIdx
    

    def growTree(self, nodeIdx):
        """
        Input:
            node : Node
        Ouput:
            Decision Tree
        """

        node = self.nodes[nodeIdx]

        if self.depth > self.maxDepth or len(node.y) <= self.minLeafSize or isHomogenous(node.y) or node.isPure():
            node.setPrediction()
            return
        


        index, threshold = split(node.X, node.y)

        # Split the data from X with index and threshold
        Xleft, Xright, yLeft, yRight = getXYSplit(node.X, node.y, index, threshold)


        node.setThreshold(threshold)
        node.setIndex(index)
        
        nodeLeft = Node(Xleft, yLeft)
        nodeRight = Node(Xright, yRight)

        nodeLeftIdx = self.addNodeLeft(nodeIdx, nodeLeft)
        nodeRightIdx = self.addNodeRight(nodeIdx, nodeRight)

        self.depth += 1
        self.growTree(nodeLeftIdx)
        self.growTree(nodeRightIdx)
        self.depth -= 1

    def predict(self, X):
        predictions = np.empty_like(X.shape[0])
        for i, x in enumerate(X):
            node = self.nodes[self.root]
            while  not node.isLeaf():
                if x[node.index] < node.threshold:
                    node = self.nodes[node.left]
                else:
                    node = self.nodes[node.right]
            predictions[i] = node.prediction
        return predictions
    
    def __str__(self):
        ret = ""

        i = 0
        for node in self.nodes:
            target_count = "["

            values, counts = np.unique(node.y, return_counts=True)

            for j in range(len(values)):
                target = values[j]
                count = counts[j]
                target_count += f"{target} : {count}, "

            target_count += "]"
            ret += f"Idx: {i} Class : {target_count} - ({node.left},{node.right}) \n"
            i += 1

        return ret



class rDecisionTree:
    def __init__(self, featureSize, minLeafSize=5, maxDepth=10):
        self.nodes = []
        self.root = 0
        self.depth = 0
        self.m = featureSize
        self.minLeafSize = minLeafSize
        self.maxDepth = maxDepth

    def fit(self, X, y):
        self.nodes.append(Node(X, y))
        # 0 is the idx of the root node
        self.growTree(0)

    def addNodeLeft(self, parentNodeIdx, node):
        newNodeIdx = len(self.nodes)
        self.nodes[parentNodeIdx].insertLeft(newNodeIdx)
        self.nodes.append(node)

        return newNodeIdx

    def addNodeRight(self, parentNodeIdx, node):
        newNodeIdx = len(self.nodes)
        self.nodes[parentNodeIdx].insertRight(newNodeIdx)
        self.nodes.append(node)

        return newNodeIdx
    

    def growTree(self, nodeIdx):
        """
        Input:
            node : Node
        Ouput:
            Decision Tree
        """

        node = self.nodes[nodeIdx]

        if self.depth > self.maxDepth or len(node.y) <= self.minLeafSize or isHomogenous(node.y) or node.isPure():
            node.setPrediction()
            return
        


        index, threshold = rSplit(node.X, node.y, self.m)

        # Split the data from X with index and threshold
        Xleft, Xright, yLeft, yRight = getXYSplit(node.X, node.y, index, threshold)


        node.setThreshold(threshold)
        node.setIndex(index)
        
        nodeLeft = Node(Xleft, yLeft)
        nodeRight = Node(Xright, yRight)

        nodeLeftIdx = self.addNodeLeft(nodeIdx, nodeLeft)
        nodeRightIdx = self.addNodeRight(nodeIdx, nodeRight)

        self.depth += 1
        self.growTree(nodeLeftIdx)
        self.growTree(nodeRightIdx)
        self.depth -= 1

    def predict(self, X):
        predictions = np.empty_like(X.shape[0])
        for i, x in enumerate(X):
            node = self.nodes[self.root]
            while  not node.isLeaf():
                if x[node.index] < node.threshold:
                    node = self.nodes[node.left]
                else:
                    node = self.nodes[node.right]
            predictions[i] = node.prediction
        return predictions
    
    def __str__(self):
        ret = ""

        i = 0
        for node in self.nodes:
            target_count = "["

            values, counts = np.unique(node.y, return_counts=True)

            for j in range(len(values)):
                target = values[j]
                count = counts[j]
                target_count += f"{target} : {count}, "

            target_count += "]"
            ret += f"Idx: {i} Class : {target_count} - ({node.left},{node.right}) \n"
            i += 1

        return ret



class Node:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.threshold = None
        self.index = None
        self.right = None
        self.left = None
        self.prediction = None

    def isLeaf(self):
        """ """
        return self.right is None and self.left is None

    def insertRight(self, nodeIdx):
        """ """
        self.right = nodeIdx

    def insertLeft(self, nodeIdx):
        """ """
        self.left = nodeIdx

    def setThreshold(self, threshold):
        """ """
        self.threshold = threshold

    def setIndex(self, index):
        """ """
        self.index = index

    def setPrediction(self):
        val,bins = np.unique(self.y, return_counts=True)
        self.prediction = val[np.argmax(bins)]

    def isPure(self):
        if len(np.unique(self.y))==1:
            return True
        else:
            cmpt = 0
            for i in range (self.X.shape[1]):
                cmpt += len(np.unique(self.X[:,i]))
            return cmpt==self.X.shape[1]

    def __str__(self):
        """ """
        return self.threshold


def giniIndex(y):
    """
    Inputs:
        y
    Output:
        float64 - Gini Index purity
    """
    _, counts = np.unique(y, return_counts=True)

    return 1 - ((counts / len(y)) ** 2).sum()


def isHomogenous(y):
    """
    Inputs:
        y: Target
    Output:
        Boolean
    """
    return len(np.unique(y)) == 1


def getYSplit(X, y, index, threshold):
    splitYRight = y[X[:, index] >= threshold]
    splitYLeft = y[X[:, index] < threshold]
    return splitYLeft, splitYRight


def getXYSplit(X, y, index, threshold):
    maskRight = X[:, index] >= threshold
    maskLeft = X[:, index] < threshold

    splitYRight = y[maskRight]
    splitYLeft = y[maskLeft]
    splitXRight = X[maskRight]
    splitXLeft = X[maskLeft]
    return splitXLeft, splitXRight, splitYLeft, splitYRight


def split(X, y):
    """ """

    bestIndice = None
    bestGini = None
    bestThreshold = None
    yLength = len(y)

    d = X.shape[1]

    for i in range(d):
        currentBestGini, currentBestThreshold = getBestSplitForOneFeature(X, y, yLength, i)
        if currentBestGini == None:
            continue
        
        if bestGini == None:
            bestGini = currentBestGini
            bestIndice = i
            bestThreshold = currentBestThreshold
        elif bestGini > currentBestGini:
            bestGini = currentBestGini
            bestIndice = i
            bestThreshold = currentBestThreshold

    return bestIndice, bestThreshold

def rSplit(X, y, m):
    """ """

    bestGini = None
    bestIndice = None
    bestThreshold = None
    yLength = len(y)

    d = X.shape[1]

    for i in np.random.choice(d, m, replace=False):
        currentBestGini, currentBestThreshold = getBestSplitForOneFeature(X, y, yLength, i)
        if currentBestGini == None:
            continue
        
        if bestGini == None:
            bestGini = currentBestGini
            bestIndice = i
            bestThreshold = currentBestThreshold
        elif bestGini > currentBestGini:
            bestGini = currentBestGini
            bestIndice = i
            bestThreshold = currentBestThreshold

    return bestIndice, bestThreshold

def getBestSplitForOneFeature(X, y, m, i):
    bestGini = None
    bestThreshold = None

    feature_values = np.unique(X[:,i])
    thresholds = (feature_values[:-1] + feature_values[1:]) / 2

    for threshold in thresholds:
        splitYLeft, splitYRight = getYSplit(X, y, i, threshold)

        giniSum = len(splitYLeft) / m * giniIndex(splitYLeft) + len(
            splitYRight
        ) / m * giniIndex(splitYRight)
        # Faut prendre en compte le cas d'égalité, utilisation d'argmin pour avoir le premier minimum qu'on a.
        if bestGini == None:
            bestGini = giniSum
            bestThreshold = threshold
        elif bestGini > giniSum:
            bestGini = giniSum
            bestThreshold = threshold
    
    return bestGini, bestThreshold




# ==========================
# rCART Tree

def rCART_tree(tree, nodeIdx, m, minLeafSize=5, maxDepth=10, depth=0):
    """
    Input:
        tree : The tree we are building
        nodeIdx : The index of the current node in the tree
        m : the number of feature we will consider per split, has to be less or equal
        minLeafSize: the number of observation minimum per leaf
        maxDepth : the maximum depth of the tree
    Ouput:
        random Decision Tree
    """
    node = tree.nodes[nodeIdx]
    assert(m <= len(node.X[0]))

    if depth > maxDepth or len(node.y) <= minLeafSize or isHomogenous(node.y) or node.isPure():
        # print("==============================")
        # print(pd.DataFrame(node.y))
        # print("==============================")
        return

    index, threshold = rSplit(node.X, node.y, m)

    # Split the data from X with index and threshold
    Xleft, Xright, yLeft, yRight = getXYSplit(node.X, node.y, index, threshold)


    node.setThreshold(threshold)
    node.setIndex(index)

    nodeLeft = Node(Xleft, yLeft)
    nodeRight = Node(Xright, yRight)

    nodeLeftIdx = tree.addNodeLeft(nodeIdx, nodeLeft)
    nodeRightIdx = tree.addNodeRight(nodeIdx, nodeRight)

    rCART_tree(tree, nodeLeftIdx, m, minLeafSize, maxDepth, depth + 1)
    rCART_tree(tree, nodeRightIdx, m, minLeafSize, maxDepth, depth + 1)


def getBootstrap(X, y, samples):
    idx =  [np.random.choice(X.shape[0], size=samples, replace=True)]
    return (X[idx], y[idx])


def randomForest(X, y, nbTrees=10, bootstrapSize= 15, featuresSize=2):
    forest = []
    for _ in range(nbTrees):
        (subX,subY) = getBootstrap(X, y, bootstrapSize)
        tree = rDecisionTree(featuresSize)
        tree.fit(subX, subY)
        forest.append(tree)
        print(tree)
    return forest