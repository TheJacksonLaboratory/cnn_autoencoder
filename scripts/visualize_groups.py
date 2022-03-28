from sklearn.manifold import TSNE, Isomap
from skimage.filters import threshold_multiotsu
import numpy as np
import zarr
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

idx = 205

z_feats = zarr.open(r'C:\Users\cervaf\Documents\Logging\segmentation_training\results\%04d_comp_feats.zarr' % idx, mode='r')
z = zarr.open(r'C:\Users\cervaf\Documents\Datasets\Kidney\Labeled_examples\%04d.zarr' % idx, mode='r')

logits = z_feats['0/0'] 
pred = sigmoid(logits[:])
feats = z_feats['1/0'] 
org = z['0/0']


h, w = org.shape[-2:]
mosaic = np.zeros([11*h, 11*w])
for j in range(11):
    for i in range(11):
        if 11*j+i >= 128: break
        mosaic[h*j:h*(j+1), w*i:w*(i+1)] = feats[0, 11*j+i, ...]

plt.subplot(1, 3, 1)
plt.imshow(logits[0, 0])
plt.title('Logits')
plt.subplot(1, 3, 2)
plt.imshow(pred[0, 0])
plt.title('Predicted probability')
plt.subplot(1, 3, 3)
plt.imshow(z['1/0'][0])
plt.title('Ground-truth')
plt.show()

plt.imshow(mosaic)
plt.show()

gray_pred = (logits - np.min(logits)) / (np.max(logits) - np.min(logits)) * 255
gray_pred = gray_pred[0, 0].astype(np.uint8)

plt.imshow(gray_pred, cmap='gray')
plt.show()

thresh = threshold_multiotsu(gray_pred, classes=5)
regions = np.digitize(gray_pred, bins=thresh)

thresh_probs = sigmoid(thresh/255 * (np.max(logits[0, 0]) - np.min(logits[0, 0])) + np.min(logits[0, 0]))
print(thresh_probs)

plt.imshow(regions, cmap=plt.cm.Spectral)
plt.show()

sample_idx = np.random.choice(regions.size, 1000, replace=False)
y_idx, x_idx = np.unravel_index(sample_idx, regions.shape)

X = feats[0, :].transpose(1, 2, 0)[(y_idx, x_idx)].astype(np.float64)

embedded_tsne = TSNE(n_components=2, learning_rate=200.0, init='random', n_jobs=4).fit_transform(X)
embedded_isomap = Isomap(n_neighbors=5, n_components=2, n_jobs=4).fit_transform(X)

plt.subplot(2, 3, 1)
plt.scatter(embedded_tsne[:, 0], embedded_tsne[:, 1], s=2, c=regions[(y_idx, x_idx)], cmap=plt.cm.Spectral)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.title('Embedded space (TSNE)', fontsize=8)
plt.subplot(2, 3, 4)
plt.scatter(embedded_isomap[:, 0], embedded_isomap[:, 1], s=2, c=regions[(y_idx, x_idx)], cmap=plt.cm.Spectral)
plt.title('Embedded space (Isomap)', fontsize=8)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.subplot(2, 3, 3)
plt.imshow(logits[0, 0], cmap=plt.cm.Spectral)
plt.scatter(x_idx, y_idx, s=1, c='black', marker='x')
plt.title('Sample positions', fontsize=8)
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(regions, cmap=plt.cm.Spectral)
plt.title('Thresholded prediction', fontsize=8)
# plt.title('Thresholded reconstruction', fontsize=8)
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(org[:].transpose(1, 2, 0))
plt.title('Original tile', fontsize=8)
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(mosaic, cmap=plt.cm.Spectral)
plt.title('Features mosaic', fontsize=8)
plt.axis('off')
plt.show()
