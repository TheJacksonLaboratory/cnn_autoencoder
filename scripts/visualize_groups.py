from sklearn.manifold import TSNE
from skimage.filters import threshold_multiotsu
import numpy as np
import zarr
import matplotlib.pyplot as plt

z_feats = zarr.open(r'C:\Users\cervaf\Documents\Logging\segmentation_training\results\0000_feats.zarr', mode='r')

pred = z_feats['0/0'] 
feats = z_feats['1/0'] 

plt.subplot(1, 2, 1)
plt.imshow(pred[0,0])
plt.subplot(1, 2, 2)
plt.imshow(1/(1 + np.exp(-pred[0,0])))
plt.show()

plt.subplot(2, 2, 1)
plt.imshow(feats[0, 0])
plt.subplot(2, 2, 2)
plt.imshow(feats[0, 1])
plt.subplot(2, 2, 3)
plt.imshow(feats[0, 2])
plt.subplot(2, 2, 4)
plt.imshow(feats[0, 3])
plt.show()

gray_pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred)) * 255
gray_pred = gray_pred[0, 0].astype(np.uint8)

plt.imshow(gray_pred, cmap='gray')
plt.show()

thresh = threshold_multiotsu(gray_pred)
regions = np.digitize(gray_pred, bins=thresh)

plt.imshow(regions, cmap='gray')
plt.show()