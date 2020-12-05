# Reference: https://www.kaggle.com/markpeng/deepinsight-efficientnet-b3-noisystudent

import math
import inspect
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

import geffnet
import cv2

import torch
from torch import nn


class DeepInsightTransformer:
    """Transform features to an image matrix using dimensionality reduction

    This class takes in data normalized between 0 and 1 and converts it to a
    CNN compatible 'image' matrix

    """

    def __init__(self,
                 feature_extractor='tsne',
                 perplexity=30,
                 pixels=100,
                 random_state=None,
                 n_jobs=None):
        """Generate an ImageTransformer instance

        Args:
            feature_extractor: string of value ('tsne', 'pca', 'kpca') or a
                class instance with method `fit_transform` that returns a
                2-dimensional array of extracted features.
            pixels: int (square matrix) or tuple of ints (height, width) that
                defines the size of the image matrix.
            random_state: int or RandomState. Determines the random number
                generator, if present, of a string defined feature_extractor.
            n_jobs: The number of parallel jobs to run for a string defined
                feature_extractor.
        """
        self.random_state = random_state
        self.n_jobs = n_jobs

        if isinstance(feature_extractor, str):
            fe = feature_extractor.casefold()
            if fe == 'tsne_exact'.casefold():
                fe = TSNE(n_components=2,
                          metric='cosine',
                          perplexity=perplexity,
                          n_iter=1000,
                          method='exact',
                          random_state=self.random_state,
                          n_jobs=self.n_jobs)
            elif fe == 'tsne'.casefold():
                fe = TSNE(n_components=2,
                          metric='cosine',
                          perplexity=perplexity,
                          n_iter=1000,
                          method='barnes_hut',
                          random_state=self.random_state,
                          n_jobs=self.n_jobs)
            elif fe == 'pca'.casefold():
                fe = PCA(n_components=2, random_state=self.random_state)
            elif fe == 'kpca'.casefold():
                fe = KernelPCA(n_components=2,
                               kernel='rbf',
                               random_state=self.random_state,
                               n_jobs=self.n_jobs)
            else:
                raise ValueError(("Feature extraction method '{}' not accepted").format(feature_extractor))
            self._fe = fe
        elif hasattr(feature_extractor, 'fit_transform') and inspect.ismethod(feature_extractor.fit_transform):
            self._fe = feature_extractor
        else:
            raise TypeError('Parameter feature_extractor is not a '
                            'string nor has method "fit_transform"')

        if isinstance(pixels, int):
            pixels = (pixels, pixels)

        # The resolution of transformed image
        self._pixels = pixels
        self._xrot = None

    def fit(self, X, y=None, plot=False):
        """Train the image transformer from the training set (X)

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
            y: Ignored. Present for continuity with scikit-learn
            plot: boolean of whether to produce a scatter plot showing the
                feature reduction, hull points, and minimum bounding rectangle

        Returns:
            self: object
        """
        # Transpose to get (n_features, n_samples)
        X = X.T

        # Perform dimensionality reduction
        x_new = self._fe.fit_transform(X)

        # Get the convex hull for the points
        chvertices = ConvexHull(x_new).vertices
        hull_points = x_new[chvertices]

        # Determine the minimum bounding rectangle
        mbr, mbr_rot = self._minimum_bounding_rectangle(hull_points)

        # Rotate the matrix
        # Save the rotated matrix in case user wants to change the pixel size
        self._xrot = np.dot(mbr_rot, x_new.T).T

        # Determine feature coordinates based on pixel dimension
        self._calculate_coords()

        # plot rotation diagram if requested
        if plot is True:
            # Create subplots
            fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=False)
            ax[0, 0].scatter(x_new[:, 0],
                             x_new[:, 1],
                             cmap=plt.cm.get_cmap("jet", 10),
                             marker="x",
                             alpha=1.0)
            ax[0, 0].fill(x_new[chvertices, 0],
                          x_new[chvertices, 1],
                          edgecolor='r',
                          fill=False)
            ax[0, 0].fill(mbr[:, 0], mbr[:, 1], edgecolor='g', fill=False)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
        return self

    @property
    def pixels(self):
        """The image matrix dimensions

        Returns:
            tuple: the image matrix dimensions (height, width)

        """
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
        """Set the image matrix dimension

        Args:
            pixels: int or tuple with the dimensions (height, width)
            of the image matrix

        """
        if isinstance(pixels, int):
            pixels = (pixels, pixels)
        self._pixels = pixels
        # recalculate coordinates if already fit
        if hasattr(self, '_coords'):
            self._calculate_coords()

    def _calculate_coords(self):
        """Calculate the matrix coordinates of each feature based on the
        pixel dimensions.
        """
        ax0_coord = np.digitize(self._xrot[:, 0],
                                bins=np.linspace(min(self._xrot[:, 0]),
                                                 max(self._xrot[:, 0]),
                                                 self._pixels[0])) - 1
        ax1_coord = np.digitize(self._xrot[:, 1],
                                bins=np.linspace(min(self._xrot[:, 1]),
                                                 max(self._xrot[:, 1]),
                                                 self._pixels[1])) - 1
        self._coords = np.stack((ax0_coord, ax1_coord))

    def transform(self, X, empty_value=0):
        """Transform the input matrix into image matrices

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
                where n_features matches the training set.
            empty_value: numeric value to fill elements where no features are
                mapped. Default = 0 (although it was 1 in the paper).

        Returns:
            A list of n_samples numpy matrices of dimensions set by
            the pixel parameter
        """

        # Group by location (x1, y1) of each feature
        # Tranpose to get (n_features, n_samples)
        img_coords = pd.DataFrame(np.vstack(
            (self._coords, X.clip(0, 1))).T).groupby(
                [0, 1],  # (x1, y1)
                as_index=False).mean()

        img_matrices = []
        blank_mat = np.zeros(self._pixels)
        if empty_value != 0:
            blank_mat[:] = empty_value
        for z in range(2, img_coords.shape[1]):
            img_matrix = blank_mat.copy()
            img_matrix[img_coords[0].astype(int),
                       img_coords[1].astype(int)] = img_coords[z]
            img_matrices.append(img_matrix)

        return img_matrices

    def fit_transform(self, X, empty_value=0):
        """Train the image transformer from the training set (X) and return
        the transformed data.

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features)
            empty_value: numeric value to fill elements where no features are
                mapped. Default = 0 (although it was 1 in the paper).

        Returns:
            A list of n_samples numpy matrices of dimensions set by
            the pixel parameter
        """
        self.fit(X)
        return self.transform(X, empty_value=empty_value)

    def feature_density_matrix(self):
        """Generate image matrix with feature counts per pixel

        Returns:
            img_matrix (ndarray): matrix with feature counts per pixel
        """
        fdmat = np.zeros(self._pixels)
        # Group by location (x1, y1) of each feature
        # Tranpose to get (n_features, n_samples)
        coord_cnt = (
            pd.DataFrame(self._coords.T).assign(count=1).groupby(
                [0, 1],  # (x1, y1)
                as_index=False).count())
        fdmat[coord_cnt[0].astype(int),
              coord_cnt[1].astype(int)] = coord_cnt['count']
        return fdmat

    @staticmethod
    def _minimum_bounding_rectangle(hull_points):
        """Find the smallest bounding rectangle for a set of points.

        Modified from JesseBuesking at https://stackoverflow.com/a/33619018
        Returns a set of points representing the corners of the bounding box.

        Args:
            hull_points : an nx2 matrix of hull coordinates

        Returns:
            (tuple): tuple containing
                coords (ndarray): coordinates of the corners of the rectangle
                rotmat (ndarray): rotation matrix to align edges of rectangle
                    to x and y
        """

        pi2 = np.pi / 2.

        # Calculate edge angles
        edges = hull_points[1:] - hull_points[:-1]
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # Find rotation matrices
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles - pi2),
            np.cos(angles + pi2),
            np.cos(angles)
        ]).T
        rotations = rotations.reshape((-1, 2, 2))

        # Apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # Find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # Find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # Return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        rotmat = rotations[best_idx]

        # Generate coordinates
        coords = np.zeros((4, 2))
        coords[0] = np.dot([x1, y2], rotmat)
        coords[1] = np.dot([x2, y2], rotmat)
        coords[2] = np.dot([x2, y1], rotmat)
        coords[3] = np.dot([x1, y1], rotmat)

        return coords, rotmat


class LogScaler:
    """Log normalize and scale data

    Log normalization and scaling procedure as described as norm-2 in the
    DeepInsight paper supplementary information.

    Note: The dimensions of input matrix is (N samples, d features)
    """

    def __init__(self):
        self._min0 = None
        self._max = None

    """
    Use this as a preprocessing step in inference mode.
    """

    def fit(self, X, y=None):
        # Min. of training set per feature
        self._min0 = X.min(axis=0)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Global max. of training set from X_norm
        self._max = X_norm.max()

    """
    For training set only.
    """

    def fit_transform(self, X, y=None):
        # Min. of training set per feature
        self._min0 = X.min(axis=0)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Global max. of training set from X_norm
        self._max = X_norm.max()

        # Normalized again by global max. of training set
        return (X_norm / self._max).clip(0, 1)

    """
    For validation and test set only.
    """

    def transform(self, X, y=None):
        # Adjust min. of each feature of X by _min0
        for i in range(X.shape[1]):
            X[:, i] = X[:, i].clip(min=self._min0[i], max=None)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Normalized again by global max. of training set
        return (X_norm / self._max).clip(0, 1)


class MoAImageSwapDataset(torch.utils.data.Dataset):
    def __init__(self,
                 features,
                 labels,
                 transformer,
                 image_size,
                 swap_prob=0.15,
                 swap_portion=0.1):
        self.features = features
        self.labels = labels
        self.transformer = transformer
        self.swap_prob = swap_prob
        self.swap_portion = swap_portion
        self.image_size = image_size

    def __getitem__(self, index):
        normalized = self.features[index, :]

        # Swap row features randomly
        normalized = self.add_swap_noise(index, normalized)

        normalized = np.expand_dims(normalized, axis=0)

        # Note: we are setting empty_value=1 to follow the setup in the paper
        image = self.transformer.transform(normalized, empty_value=1)[0]

        # Resize to target size
        gene_cht = cv2.resize(image, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_CUBIC)

        # Convert to 3 channels
        image = np.repeat(gene_cht[np.newaxis, :, :], 3, axis=0)

        return image, self.labels[index, :]

    def add_swap_noise(self, index, X):
        if np.random.rand() < self.swap_prob:
            swap_index = np.random.randint(self.features.shape[0], size=1)[0]
            # Select only gene expression and cell viability features
            swap_features = np.random.choice(
                np.array(range(3, self.features.shape[1])),
                size=int(self.features.shape[1] * self.swap_portion),
                replace=False)
            X[swap_features] = self.features[swap_index, swap_features]

        return X

    def __len__(self):
        return self.features.shape[0]


class MoAImageDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transformer, image_size):
        self.features = features
        self.labels = labels
        self.transformer = transformer
        self.image_size = image_size

    def __getitem__(self, index):
        normalized = self.features[index, :]
        normalized = np.expand_dims(normalized, axis=0)

        # Note: we are setting empty_value=1 to follow the setup in the paper
        image = self.transformer.transform(normalized, empty_value=1)[0]

        # Resize to target size
        gene_cht = cv2.resize(image, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_CUBIC)

        # Convert to 3 channels
        image = np.repeat(gene_cht[np.newaxis, :, :], 3, axis=0)

        return image, self.labels[index, :]

    def __len__(self):
        return self.features.shape[0]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transformer, image_size):
        self.features = features
        self.labels = labels
        self.transformer = transformer
        self.image_size = image_size

    def __getitem__(self, index):
        normalized = self.features[index, :]
        normalized = np.expand_dims(normalized, axis=0)

        # Note: we are setting empty_value=1 to follow the setup in the paper
        image = self.transformer.transform(normalized, empty_value=1)[0]

        # Resize to target size
        gene_cht = cv2.resize(image, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_CUBIC)

        # Convert to 3 channels
        image = np.repeat(gene_cht[np.newaxis, :, :], 3, axis=0)

        return image, -1

    def __len__(self):
        return self.features.shape[0]


def initialize_weight_goog(m, n='', fix_group_fanout=True):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


class MoAEfficientNet(nn.Module):
    def __init__(
            self,
            pretrained_model_name,
            num_classes=206,
            in_chans=3,
            drop_rate=0.,
            drop_connect_rate=0.,
            fc_size=512,
            weight_init='goog'):
        super(MoAEfficientNet, self).__init__()

        self.backbone = getattr(geffnet, pretrained_model_name)(
            pretrained=True,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_connect_rate=drop_connect_rate,
            weight_init=weight_init)

        self.backbone.classifier = nn.Sequential(
            nn.Linear(
                self.backbone.classifier.in_features,
                fc_size,
                bias=True
            ),
            nn.ELU(),
            nn.Linear(fc_size, num_classes, bias=True)
        )

        if self.training:
            for m in self.backbone.classifier.modules():
                initialize_weight_goog(m)

    def forward(self, x):
        return self.backbone(x)
