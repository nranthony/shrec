from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class RigidTransform(BaseEstimator, TransformerMixin):
    """
    Compute the rigid transformation (rotation, translation, scaling) aligning two
    sets of points using the Kabsch-Umeyama algorithm. The first argument is transformed
    to align with the second argument.

    Attributes:
        scale (float): scaling factor
        rotation (array): rotation matrix
        translation (array): translation vector
    """

    def __init__(self, scale=None, rotation=None, translation=None):
        self.scale = scale
        self.rotation = rotation
        self.translation = translation

    def fit(self, X, y=None):
        """
        Fit transformation matrices that align X to y.

        Args:
            X: (N, M) array of points to be transformed
            y: (N, M) array of points to align with
        """
        b, a = X, y
        assert a.shape == b.shape
        n, m = a.shape

        ca = np.mean(a, axis=0)
        cb = np.mean(b, axis=0)

        var_a = np.mean(np.linalg.norm(a - ca, axis=1) ** 2)

        H = ((a - ca).T @ (b - cb)) / n
        U, D, VT = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
        S = np.diag([1] * (m - 1) + [d])

        R = U @ S @ VT
        c = var_a / np.trace(np.diag(D) @ S)
        t = ca - c * R @ cb

        transform_params = (R, c, t)

        self.rotation = R
        self.scale = c
        self.translation = t

    def fit_transform(self, X, y=None):
        """
        Fit transformation matrices that align X to y and apply them to X.

        Args:
            X: (N, M) array of points to be transformed
            y: (N, M) array of points to align with
        """
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        X_transformed = self.scale * self.rotation @ X.T + self.translation[:, np.newaxis]
        return X_transformed.T

    def inverse_transform(self, X):
        X = X.T
        X = (X - self.translation[:, np.newaxis]) / self.scale
        X = self.rotation.T @ X
        return X.T
