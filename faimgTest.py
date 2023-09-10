#%%
# imports
import time

from sklearn.datasets import load_sample_image
import faimg as fg
import numpy as np
import random
import matplotlib.pyplot as plt

#%%
# Tests for Image class:
china = load_sample_image("china.jpg")
imgProc = fg.ImageProcessor(china)

# %%
# Gini index

y = [1, 1, 1, 1, 2, 32, 3, 12, 312, 31, 231, 23]
print(len(np.bincount(y)))
gini = fg.giniIndex(y)
print(gini)

#%%
# Extracts patches speed test
start = time.time()
all_patches = imgProc.extract_all_patches(10, 3, 2)
end = time.time()
print("Time consumed in working: ", end - start)
print(np.array(all_patches).shape)

start_np = time.time()
all_patches_np = imgProc.extract_all_patches_np(10, 3, 2)
end_np = time.time()
print("Time consumed in working: ", end_np - start_np)
print((all_patches_np == all_patches).all())
print(all_patches_np.shape)

#%%
# Extract regions
region_size = 3
patch = fg.Patch(china[0:20, 0:20], region_size)

# region_positions = np.array([[0,0], [9,7]])

regions = patch.retrieve_random_regions()

print(regions[0])


#%% Extract regions and operators
patch_x = 150
patch_y = 150
patch_size = 20
region_size = 3
region_positions = np.array([[0, 0], [9, 7]])

patch = fg.Patch(
    china[patch_x : patch_x + patch_size, patch_y : patch_y + patch_size], region_size
)

regions = patch.retrieve_regions(region_positions)

print("Region : ")
print(regions[0])

print("Span : ")
span = patch.get_span(regions[0])
print(span)

for op in fg.RegionOperator:
    print(op)
    print(patch.apply_operator(regions[0], op))

#%% Referential value (used when 1 regions for distance) :

patch = fg.Patch(np.array([[[1, 1, 1]]]), 0)

reference_value = patch.get_random_reference_value()

print(reference_value)
print([patch])


#%% Random regions
patchSize = 20
regionSize = 5


regions = fg.getRandomRegionsPosition(patchSize, regionSize)
bar = random.choice(list(fg.RegionOperator))
print(bar)
print("regions : ", regions)

#%% Predictions playground

prediction = np.array([[1, 1, 1], [2, 2, 1], [2, 2, 2]])

print(prediction)
print(prediction.T)


def pred(array):
    counts = np.bincount(array)
    return np.argmax(counts)


print(pred(prediction.T[0]))


#%% Compute distance
patch_x = 150
patch_y = 150
patch_size = 20
region_size = 3
operator = fg.RegionOperator.MAX_SPAN
patch = fg.Patch(
    china[patch_x : patch_x + patch_size, patch_y : patch_y + patch_size], region_size
)

# 1 region
region_positions = np.array([[0, 0]])

res = patch.compute_feature(region_positions, operator)
print("Result type : ", res.dtype)
print("For 1 region : ", res)
# 2 region
region_positions = np.array([[0, 0], [4, 15]])

res = patch.compute_feature(region_positions, operator)
print("For 2 regions : ", res)

# 4 region
region_positions = np.array([[1, 1], [4, 15], [16, 2], [7, 9]])

res = patch.compute_feature(region_positions, operator)
print("For 4 regions : ", res)


#%% Loading utils
from PIL import Image


def loadDataAndLabel(imgDataPath, labelDataPath):
    imData = Image.open(imgDataPath)
    imLabel = Image.open(labelDataPath)

    X = np.array(imData)
    yRaw = np.array(imLabel)

    class_mapping = {
        (0x3C, 0x10, 0x98): 1,  # Building
        (0x84, 0x29, 0xF6): 2,  # Land (unpaved area)
        (0x6E, 0xC1, 0xE4): 3,  # Road
        (0xFE, 0xDD, 0x3A): 4,  # Vegetation
        (0xE2, 0xA9, 0x29): 5,  # Water
        (0x9B, 0x9B, 0x9B): 6,  # Unlabeled
    }

    def rgb_to_labels(image, class_mapping):
        # Initialize an empty array to store the labels
        labels = np.empty(image.shape[:2], dtype=np.uint8)

        # Iterate over each pixel in the image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Get the RGB value of the current pixel
                rgb = tuple(image[i, j])

                # Map the RGB value to a class label using the provided mapping
                label = class_mapping.get(
                    rgb, 6
                )  # Default to class 6 - Unlabeled if no mapping exists

                # Assign the label to the corresponding pixel in the labels array
                labels[i, j] = label

        return labels

    y = rgb_to_labels(yRaw, class_mapping)

    return X, y


#%% Loading the images
# - Tile 1 - part 005
patchSize = 20
regionSize = 5


print("Loading img...")
start = time.time()
# Images from https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery
X, y = loadDataAndLabel("data/image_part_005_data.jpg", "data/image_part_005_label.png")
end = time.time()
print("Image loaded in ", end - start, "secs !")
print(X.shape)
print(y.shape)

X = X[0:125, 0:125]
y = y[0:125, 0:125]

print("Processing patches ...")
imProc = fg.ImageProcessor(X)
patches = imProc.extract_all_patches_np(patchSize, regionSize)

yWithPatch = y[
    patchSize // 2 : y.shape[0] - patchSize // 2 + 1,
    patchSize // 2 : y.shape[1] - patchSize // 2 + 1,
]

y = yWithPatch.flatten()

print("Patches processed ! shape : ", patches.shape, y.shape)
assert patches.shape == y.shape

#%%
# Testing Hansch tree fit on loaded images
print("Creating Hansch Tree ...")
start = time.time()
tree = fg.HanschTree(patchSize, regionSize, maxDepth=2, minSampleSize=10)
tree.fit(patches, y)
end = time.time()
print("Tree created in : ", end - start, "secs !")

#%%
# Print tree
print(tree)

#%%
# Testing Hansch tree predict on images
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier()
dummy.fit(patches, y)
dummy_predict = dummy.predict(patches)

prediction = tree.predict(patches)
print(dummy_predict.shape)
print(prediction.shape)
print("dummy Accuracy : ", np.count_nonzero(y == dummy_predict) / y.shape[0])
print("tree Accuracy : ", np.count_nonzero(y == prediction) / y.shape[0])


#%%
# Testing Hansch forest fit on loaded images
print("Creating Hansch Forest ...")
start = time.time()
forest = fg.HanschForest(patchSize=patchSize, regionSize=regionSize, nbTrees=5, maxDepth=3, minSampleSize=10)
forest.fit(patches, y)
end = time.time()
print("forest created in : ", end - start, "secs !")


#%%
# Testing Hansch forest predict on images
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier()
dummy.fit(patches, y)
dummy_predict = dummy.predict(patches)

prediction = forest.predict(patches)
print(dummy_predict.shape)
print(prediction.shape)
print("dummy Accuracy : ", np.count_nonzero(y == dummy_predict) / y.shape[0])
print("forest Accuracy : ", np.count_nonzero(y == prediction) / y.shape[0])


#%%
# Extracts patches
patch_size = 50
patch = imgProc.retrieve_random_image_patch(patch_size)
assert patch.shape[:2] == (patch_size, patch_size)

regions = imgProc.retrieve_regions(patch)
assert len(regions) >= 1 and len(regions) <= 4
for region in regions:
    assert region.shape[0] >= 10 and region.shape[1] >= 10
plt.imshow(china)
plt.figure()
plt.imshow(patch)
plt.axis("off")

for region in regions:
    plt.figure()
    plt.imshow(region)
    plt.axis("off")
