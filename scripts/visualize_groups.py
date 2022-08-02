import math
from tqdm import tqdm
import sympy.geometry as geom
from sklearn.manifold import TSNE, Isomap
from skimage.filters import threshold_multiotsu
import numpy as np
import zarr

from matplotlib.backend_bases import MouseButton
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


positions = []

fig, ax = plt.subplots()
ax.scatter(embedded_tsne[:, 0], embedded_tsne[:, 1], s=2, c=regions[(y_idx, x_idx)], cmap=plt.cm.Spectral)

def on_click(event):
    if event.button is MouseButton.RIGHT:
        print('disconnecting callback')

        inv = ax.transData.inverted()
        inv_xy_last = positions[-1]
        inv_xy_first = positions[0]

        ax.plot((inv_xy_last[0], inv_xy_first[0]), (inv_xy_last[1], inv_xy_first[1]), 'b:')

        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.disconnect(binding_id)
        
    if event.button is MouseButton.LEFT:
        x, y = event.x, event.y
        
        inv = ax.transData.inverted()
        inv_xy_new = inv.transform((x, y))

        if len(positions) > 0:
            inv_xy_last = positions[-1]
            print(inv_xy_new, inv_xy_last)
            ax.plot((inv_xy_last[0], inv_xy_new[0]), (inv_xy_last[1], inv_xy_new[1]), 'b:')

            fig.canvas.draw()
            fig.canvas.flush_events()

        positions.append(inv_xy_new)

binding_id = plt.connect('button_press_event', on_click)
plt.show()

positions = np.array(positions)
plt.scatter(embedded_tsne[:, 0], embedded_tsne[:, 1], s=2, cmap=plt.cm.Spectral)
plt.plot(positions[:, 0], positions[:, 1], 'b:')
plt.plot((positions[-1, 0], positions[0, 0]), (positions[-1, 1], positions[0, 1]), 'b:')
plt.show()

selection_vertices = [geom.Point2D(x, y) for x, y in positions]
selection = geom.Polygon(*selection_vertices)
selection_area = math.fabs(selection.area)

bb_tl = (np.min(positions[:, 0]), np.min(positions[:, 1]))
bb_br = (np.max(positions[:, 0]), np.max(positions[:, 1]))

bb_indices = np.nonzero(
    np.bitwise_and(
        np.bitwise_and(bb_tl[0] <= embedded_tsne[:, 0], embedded_tsne[:, 0] <= bb_br[0]), 
        np.bitwise_and(bb_tl[1] <= embedded_tsne[:, 1], embedded_tsne[:, 1] <= bb_br[1])
    )
)[0]

embedded_points = [geom.Point2D(x, y) for x, y in embedded_tsne[bb_indices]]

q = tqdm(total=len(embedded_points))
selected_points = []
for i, pt in zip(bb_indices, embedded_points):
    areas = [math.fabs(geom.Triangle(p1, p2, pt).area) for p1, p2 in zip(selection_vertices[:-1], selection_vertices[1:])]
    areas.append(math.fabs(geom.Triangle(selection_vertices[-1], selection_vertices[0], pt).area))
    areas = sum(areas)

    if areas == selection_area:
        selected_points.append(i)

    q.set_description_str('Current point %s in the selection' % ('is' if areas == selection_area else 'is not'))
    q.update()

q.close()

len(selected_points)


plt.subplot(2, 3, 1)
plt.scatter(embedded_tsne[selected_points, 0], embedded_tsne[selected_points, 1], s=2, c=regions[(y_idx[selected_points], x_idx[selected_points])], cmap=plt.cm.Spectral)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.title('Embedded space (TSNE)', fontsize=8)
plt.subplot(2, 3, 4)
plt.scatter(embedded_isomap[selected_points, 0], embedded_isomap[selected_points, 1], s=2, c=regions[(y_idx[selected_points], x_idx[selected_points])], cmap=plt.cm.Spectral)
plt.title('Embedded space (Isomap)', fontsize=8)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.subplot(2, 3, 3)
plt.imshow(logits[0, 0], cmap=plt.cm.Spectral)
plt.scatter(x_idx[selected_points], y_idx[selected_points], s=1.5, c='black', marker='x')
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
plt.show()
