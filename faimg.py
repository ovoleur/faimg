import multiprocessing
import numpy as np
import random
from numpy.random import default_rng
from faimg_rs import get_best_split

rng = default_rng()

class ImageProcessor:
    def __init__(self, img):
        self.image = img

    def retrieve_random_image_patch(self, crop_size):
        """
        inputs:
            -crop_size : The size of the portion extracted from the original image (Also noted w̃).
        output:
            -image_patch : A portion of the original image of size w̃*w̃
        """
        height, width = self.image.shape[:2]

        max_left = width - crop_size
        max_top = height - crop_size

        assert max_left >= 0 and max_top >= 0, "Crop size is too large for the image."

        left = np.random.randint(0, max_left + 1)
        top = np.random.randint(0, max_top + 1)

        image_patch = self.image[top : top + crop_size, left : left + crop_size]

        return image_patch

    def extract_all_patches(self, patchSize, regionSize, stride):
        num_rows, num_cols = self.image.shape[:2]
        patches = []

        for i in range(num_rows - patchSize + 1):
            for j in range(num_cols - patchSize + 1):
                patch = self.image[i : i + patchSize, j : j + patchSize]
                patches.append(Patch(patch, regionSize))

        return np.array(patches[::stride, ::stride])

    # Faster than extract_all_patches when patchsize is small
    def extract_all_patches_np(self, patchSize, regionSize, stride):
        dims = self.image.shape[2]
        patches = np.lib.stride_tricks.sliding_window_view(
            np.array(self.image[::stride, ::stride]), (patchSize, patchSize, dims)
        )
        patches = patches.reshape(-1, patchSize, patchSize, dims)

        # return patches
        patches_object = np.empty(patches.shape[0], dtype=object)

        for i, patch in enumerate(patches):
            patches_object[i] = Patch(patch, regionSize)

        return patches_object


from enum import Enum


class RegionOperator(Enum):
    CENTER_PIXEL = 1
    MEAN_PIXEL = 2
    MIN_SPAN = 3
    MAX_SPAN = 4
    MEDIAN_SPAN = 5



class Patch:
    def __init__(self, patch, region_size):
        assert patch.ndim == 3
        assert patch.shape[0] > region_size

        self.img = patch
        self.region_size = region_size
        self.dim = patch.shape[2]

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return str(self.img)

    def retrieve_random_regions(self):
        nbRegions = random.choice([1, 2, 4])
        regions = [
            random.randint(10, min(self.img.shape[:2])) for i in range(nbRegions)
        ]
        return regions

    def retrieve_regions(self, region_positions):

        regions = []
        for [x, y] in region_positions:
            regions.append(self.img[x : x + self.region_size, y : y + self.region_size])

        return regions

    def get_random_reference_value(self):
        return np.random.randint(256, size=self.dim)

    def central_pixel(self, region):
        return region[region.shape[0] // 2, region.shape[1] // 2]

    def mean_pixel(self, region):
        return region.mean(axis=(0, 1))

    # A span for a pixel is the sum of all values over all dimensions.
    # For rgb images it's simply the sum of r + g + b
    def get_span(self, region):
        return region.sum(axis=-1)

    def median_span_pixel(self, region):
        span = self.get_span(region)

        # If multiple spans meet the condition, we take the first one
        return region[np.median(span, axis=(0, 1)) == span][0]

    def max_span_pixel(self, region):
        span = self.get_span(region)

        # If multiple spans meet the condition, we take the first one
        return region[np.max(span, axis=(0, 1)) == span][0]

    def min_span_pixel(self, region):
        span = self.get_span(region)
        # If multiple spans meet the condition, we take the first one
        return region[np.min(span, axis=(0, 1)) == span][0]

    def apply_operator(self, region, operator: RegionOperator):
        if operator == RegionOperator.CENTER_PIXEL:
            return self.central_pixel(region)
        elif operator == RegionOperator.MEAN_PIXEL:
            return self.mean_pixel(region)
        elif operator == RegionOperator.MAX_SPAN:
            return self.max_span_pixel(region)
        elif operator == RegionOperator.MIN_SPAN:
            return self.min_span_pixel(region)
        elif operator == RegionOperator.MEDIAN_SPAN:
            return self.median_span_pixel(region)
        else:
            raise Exception("Trying to apply an illegal operator")

    def euclidian_distance(self, pixel1, pixel2):
        return np.linalg.norm(pixel1 - pixel2)

    def compute_feature(self, region_positions, operator: RegionOperator):
        regions = self.retrieve_regions(region_positions)
        pixels_extracted = [self.apply_operator(r, operator) for r in regions]
        if len(regions) == 1:
            return self.euclidian_distance(
                pixels_extracted[0], self.get_random_reference_value()
            )
        elif len(regions) == 2:
            return self.euclidian_distance(pixels_extracted[0], pixels_extracted[1])
        else:
            return self.euclidian_distance(
                pixels_extracted[0], pixels_extracted[1]
            ) - self.euclidian_distance(pixels_extracted[2], pixels_extracted[3])


def getRandomRegionsPosition(patchSize, regionSize):
    nbRegions = random.choice([1, 2, 4])

    # Implementation not so good, but guarantees that two regions will not be the same.
    # This is ok to not be super optimized at first as this is not the hotspot in the algo
    regionsPositions = np.zeros((nbRegions, 2), dtype=np.int64)
    idx = 0
    while idx != nbRegions:
        try_add = np.random.randint(patchSize - regionSize, size=2)

        can_add = True
        for i in range(idx):
            if (regionsPositions[i] == try_add).all():
                can_add = False
        
        if can_add:
            regionsPositions[idx] = try_add
            idx += 1




    return regionsPositions




class HanschForest:
    def __init__(
        self, featureGenerated=20, patchSize=10, regionSize=3, nbTrees=5, minSampleSize=1, maxDepth=None
    ):
        self.forest = np.empty(nbTrees, dtype=object)
        self.patches = None
        self.patchSize = patchSize
        self.regionSize = regionSize
        self.nbTrees = nbTrees
        self.minSampleSize = minSampleSize
        self.maxDepth = maxDepth
        self.featureGenerated = featureGenerated

    def fit(self, patches, y):
        self.nbClass_ = len(np.bincount(y))
        self.patches = patches


        cpu_count = multiprocessing.cpu_count()
        manager = multiprocessing.Manager()
        tree_dict = manager.dict()

        progress = 0

        handles = []

        for idx in range(self.nbTrees):
            create_tree_process = multiprocessing.Process(
                target=self.fit_tree_job, args=(tree_dict, idx, patches, y)
            )


            create_tree_process.start()
            handles.append(create_tree_process)

            if len(handles) >= cpu_count:
                for handle in handles:
                    progress += 1
                    handle.join()

                print(f"{progress}/{self.nbTrees} tree created")
                handles = []


        for handle in handles:
            handle.join()

        print(f"forest created !")
        for idx in range(self.nbTrees):
            self.forest[idx] = tree_dict[idx]

    def fit_tree_job(self, return_dict, idx, patches, y):
        tree = HanschTree(
            patchSize=self.patchSize, 
            featureGenerated=self.featureGenerated,
            regionSize=self.regionSize, 
            minSampleSize=self.minSampleSize, 
            maxDepth=self.maxDepth,
            nbClass=self.nbClass_
        )
        tree.fit(patches, y)
        return_dict[idx] = tree

    def predict(self, patches):
        # FIXME: Implentation not so good
        patchesLen = patches.shape[0]


        # Start all predictions processes
        handles = []
        manager = multiprocessing.Manager()
        prediction_dict = manager.dict()

        for idx, tree in enumerate(self.forest):
            prediction_process = multiprocessing.Process(
                target=self.inner_predict, args=(idx, tree, prediction_dict, patches)
            )
            prediction_process.start()
            handles.append(prediction_process)

        # Wait the end of all processes
        for handle in handles:
            handle.join()

        forestPredictions = np.empty((self.nbTrees, patchesLen), dtype=np.int64)


        # Collect result from processes
        for idx in range(self.nbTrees):
            forestPredictions[idx] = prediction_dict[idx]

        predictions = forestPredictions.T

        ret = np.empty(patchesLen, dtype=np.int64)

        for idx, prediction in enumerate(predictions):
            # For each response from all the trees, we keep the one with the most vote
            counts = np.bincount(prediction)
            ret[idx] = np.argmax(counts)

        return ret

    def inner_predict(self, idx, tree, prediction_dict, patches):
        prediction_dict[idx] = tree.predict(patches)


class HanschTree:
    def __init__(self, featureGenerated=20, patchSize=10, regionSize=3, minSampleSize=3, maxDepth=20, nbClass=None):
        self.nodes = []
        self.root = 0
        self.depth = 0
        self.patchSize = patchSize
        self.regionSize = regionSize
        self.minSampleSize = minSampleSize
        self.maxDepth = maxDepth
        self.nbClass = nbClass
        self.featureGenerated = featureGenerated

    def addNodeLeft(self, nodeIdx, node):
        newNodeIdx = len(self.nodes)
        self.nodes[nodeIdx].insertLeft(newNodeIdx)
        self.nodes.append(node)

        return newNodeIdx

    def addNodeRight(self, nodeIdx, node):
        newNodeIdx = len(self.nodes)
        self.nodes[nodeIdx].insertRight(newNodeIdx)
        self.nodes.append(node)

        return newNodeIdx

    def fit(self, patches, y):
        self.nodes.append(HanschNode(patches, y))
        self.growTree(self.root)

    def growTree(self, nodeIdx):
        node = self.nodes[nodeIdx]

        if (
            (self.maxDepth != None and self.depth > self.maxDepth)
            or len(node.y) <= self.minSampleSize
            or node.isPure()
        ):
            node.setPrediction()
            return

        best_gini, best_split_data, best_split = None, None, None

        for _ in range(self.featureGenerated):
            regionsPosition = getRandomRegionsPosition(self.patchSize, self.regionSize)
            operator = random.choice(list(RegionOperator))

            split = HanschSplit(self.nbClass ,operator, regionsPosition)

            split_res = split.getPatchAndLabelSplit(node.patches, node.y)

            if split_res == None:
                continue

            gini, split_data = split_res

            if best_gini == None or best_gini > gini:
                best_gini = gini
                best_split_data = split_data
                best_split = split

        # We didn't find any possible split, can be due to patch with same pixel value
        if best_split_data == None:
            node.setPrediction()
            return

        patchesleft, patchesRight, yLeft, yRight = best_split_data

        node.setSplit(best_split)

        nodeLeft = HanschNode(patchesleft, yLeft)
        nodeRight = HanschNode(patchesRight, yRight)

        nodeLeftIdx = self.addNodeLeft(nodeIdx, nodeLeft)
        nodeRightIdx = self.addNodeRight(nodeIdx, nodeRight)

        self.depth += 1
        self.growTree(nodeLeftIdx)
        self.growTree(nodeRightIdx)
        self.depth -= 1

    def predict(self, patches):
        predictions = np.empty(patches.shape[0], dtype=np.int64)
        for idx, patch in enumerate(patches):
            node = self.nodes[self.root]
            while not node.isLeaf():
                split = node.split
                value = patch.compute_feature(split.regionsPosition, split.operator)

                if value < split.threshold:
                    node = self.nodes[node.left]
                else:
                    node = self.nodes[node.right]

            predictions[idx] = node.prediction
        return predictions

    def __str__(self):
        ret = ""

        for i, node in enumerate(self.nodes):
            target_count = "["

            values, counts = np.unique(node.y, return_counts=True)

            for j in range(len(values)):
                target = values[j]
                count = counts[j]
                target_count += f"{target} : {count}, "

            target_count += "]"
            ret += f"Idx: {i} : {target_count}"
            if node.isLeaf():
                ret += f" prediction {node.prediction} \n"
            else:
                ret += f" childs ({node.left},{node.right}) \n"
                ret += f"\tSplit op : {node.split.operator} | threshold : {node.split.threshold}"
                ret += f" | position : \n{node.split.regionsPosition}\n"

        return ret
    
    def __repr__(self):
        ret = ""

        for i, node in enumerate(self.nodes):
            target_count = "["

            values, counts = np.unique(node.y, return_counts=True)

            for j in range(len(values)):
                target = values[j]
                count = counts[j]
                target_count += f"{target} : {count}, "

            target_count += "]"
            ret += f"Idx: {i} : {target_count}"
            if node.isLeaf():
                ret += f" prediction {node.prediction} \n"
            else:
                ret += f" childs ({node.left},{node.right}) \n"
                ret += f"\tSplit op : {node.split.operator} | threshold : {node.split.threshold}"
                ret += f" | position : \n{node.split.regionsPosition}\n"

        return ret


class HanschSplit:
    def __init__(self, nbClass, operator: RegionOperator, regionsPosition):
        # FIXME: One Operator or one per region ?
        self.operator = operator

        self.regionsPosition = regionsPosition
        self.threshold = None
        self.nbClass = nbClass

    # Returns : leftPatches, rightPatches, leftLabels, rightLabels
    def getPatchAndLabelSplit(self, patches, y):
        # lower case x cause its a vector
        x = np.empty(patches.shape[0])

        for idx, patch in enumerate(patches):
            x[idx] = patch.compute_feature(self.regionsPosition, self.operator)

        return self.findBestSplitRs(patches, x, y)
        # return self.findBestSplit(patches, x, y)

    def findBestSplitRs(self, patches, x, y):
        feature_values = np.unique(x)
        thresholds = (feature_values[:-1] + feature_values[1:]) / 2

        if thresholds.shape[0] == 0:
            self.threshold = None
            return None


        gini, threshold = get_best_split(x, y, self.nbClass, thresholds)

        self.threshold = threshold

        return (gini, self.getXYSplit(patches, x, y, self.threshold))
    
    def findBestSplit(self, patches, x, y):
        bestGini = None

        feature_values = np.unique(x)
        thresholds = (feature_values[:-1] + feature_values[1:]) / 2

        m = y.shape[0]

        for threshold in thresholds:
            splitYLeft, splitYRight = self.getYSplit(x, y, threshold)

            giniSum = len(splitYLeft) / m * giniIndex(splitYLeft) + len(
                splitYRight
            ) / m * giniIndex(splitYRight)

            if bestGini == None:
                bestGini = giniSum
                self.threshold = threshold
            elif bestGini > giniSum:
                bestGini = giniSum
                self.threshold = threshold

        if self.threshold == None:
            return None

        return (bestGini, self.getXYSplit(patches, x, y, self.threshold))

    def getSplitMask(self, x, threshold):
        return x < threshold, x >= threshold

    def getXYSplit(self, patches, x, y, threshold):
        leftMask, rightMask = self.getSplitMask(x, threshold)

        return patches[leftMask], patches[rightMask], y[leftMask], y[rightMask]

    def getYSplit(self, x, y, threshold):
        leftMask, rightMask = self.getSplitMask(x, threshold)
        return y[leftMask], y[rightMask]


class HanschNode:
    def __init__(self, patches, y):
        self.patches = patches
        self.y = y
        self.split = None
        self.right = None
        self.left = None
        self.prediction = None

    def isLeaf(self):
        return self.prediction is not None

    def insertRight(self, nodeIdx):
        self.right = nodeIdx

    def insertLeft(self, nodeIdx):
        self.left = nodeIdx

    def setSplit(self, split: HanschSplit):
        self.split = split

    def setPrediction(self):
        """
        Set the prediction to be the class that occurs the most
        """
        val, bins = np.unique(self.y, return_counts=True)
        self.prediction = val[np.argmax(bins)]

    def isPure(self):
        if len(np.unique(self.y)) == 1:
            return True
        else:
            return False

    def __str__(self):
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
