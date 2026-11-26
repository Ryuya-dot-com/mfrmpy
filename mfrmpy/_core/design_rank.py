"""Row-space tests after exact elimination of one-hot person columns.

The reduced facet/constraint system still uses a bounded dense SVD. No graph
connectivity or iterative-solver convergence is substituted for numeric rank.
"""
import numpy as np
import pandas as pd
from scipy import sparse


MAX_DENSE_CELLS = 5_000_000


class RowSpace:
    """Eliminate person pivots once and reuse the remaining row space."""

    def __init__(self, matrix, n_person=0, *, max_dense_cells=MAX_DENSE_CELLS):
        matrix = sparse.csr_matrix(matrix, dtype=float)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        self.shape = matrix.shape
        personal = matrix[:, :n_person]
        candidates = np.flatnonzero(np.diff(personal.indptr) == 1)
        columns = personal.indices[personal.indptr[candidates]]
        self.pivots, first = np.unique(columns, return_index=True)
        rows = candidates[first]
        values = personal.data[personal.indptr[rows]]
        self.remaining = np.setdiff1d(np.arange(matrix.shape[1]), self.pivots)
        self.pivot_rows = matrix[rows][:, self.remaining].multiply(1. / values[:, None]).tocsr()
        reduced = (matrix[:, self.remaining] - matrix[:, self.pivots] @ self.pivot_rows).tocsr()
        reduced.eliminate_zeros()
        reduced.sort_indices()
        # Repeated within-person assignment differences carry the same rank.
        unique = {}
        for i in np.flatnonzero(np.diff(reduced.indptr)):
            a, b = reduced.indptr[i:i + 2]
            key = (reduced.indices[a:b].tobytes(), reduced.data[a:b].tobytes())
            unique.setdefault(key, i)
        reduced = reduced[list(unique.values())]
        self.active = np.flatnonzero(reduced.getnnz(axis=0))
        self.inactive = np.setdiff1d(np.arange(len(self.remaining)), self.active)
        reduced = reduced[:, self.active]
        self.reduced_shape = reduced.shape
        # Both decisions concern the reduced row space. Repeating identical
        # observations must not loosen the tolerance for estimability.
        # Leave a small roundoff allowance for SVD and the two projections in
        # residual(); this depends on the reduced system, never the row count.
        self.tolerance = 8 * max(1, *reduced.shape) * np.finfo(float).eps
        self.available = (int(reduced.shape[0]) * int(reduced.shape[1]) <= max_dense_cells)
        self.rank = None
        self.basis = None
        if not self.available:
            return
        dense = reduced.toarray()
        _, singular, vh = np.linalg.svd(dense, full_matrices=False)
        tol = self.tolerance
        self.basis = vh[singular > tol * singular[0]] if len(singular) else vh
        self.rank = len(self.pivots) + len(self.basis)

    def residual(self, target):
        """Represent a target in the quotient by the audited row space."""
        if not self.available:
            return None
        row = sparse.csr_matrix(target, dtype=float)
        if row.shape != (1, self.shape[1]):
            raise ValueError("A target must have one coefficient per design column")
        reduced = row[:, self.remaining] - row[:, self.pivots] @ self.pivot_rows
        value = reduced.toarray().ravel()
        active = value[self.active]
        value[self.active] = active - (active @ self.basis.T) @ self.basis
        return value

    def contains(self, target):
        residual = self.residual(target)
        if residual is None:
            return None
        norm = sparse.linalg.norm(sparse.csr_matrix(target))
        return bool(np.linalg.norm(residual) <= self.tolerance * max(1., norm))


def incidence_design(data, facets, levels, signs, constraints=()):
    """Sparse signed incidence, followed by homogeneous constraint rows."""
    offsets = np.cumsum([0, *[len(levels[f]) for f in facets]])
    row = []
    col = []
    value = []
    for f, start in zip(facets, offsets):
        lookup = {level: i for i, level in enumerate(levels[f])}
        indices = data[f].map(lookup).to_numpy(dtype=int)
        row.extend(range(len(data)))
        col.extend(start + indices)
        value.extend([signs[f]] * len(data))
    for i, indices in enumerate(constraints, len(data)):
        indices = list(indices)
        row.extend([i] * len(indices))
        col.extend(indices)
        value.extend([1.] * len(indices))
    return sparse.csr_matrix((value, (row, col)), shape=(len(data) + len(constraints), int(offsets[-1])))


def constraint_indices(specs):
    """Homogeneous equations for the engine's fixed/group/centering specs."""
    rows = []
    offset = 0
    for spec in specs:
        anchors = np.asarray(spec["anchors"])
        free = np.flatnonzero(np.isnan(anchors))
        rows.extend([[offset + i] for i in np.flatnonzero(np.isfinite(anchors))])
        groups = spec["groups"]
        grouped = set()
        for group in {groups[i] for i in free if not pd.isna(groups[i]) and groups[i] != ""}:
            indices = [i for i in free if groups[i] == group]
            rows.append([offset + i for i in indices])
            grouped.update(indices)
        other = [i for i in free if i not in grouped]
        if spec["centered"] and other:
            rows.append([offset + i for i in other])
        offset += len(anchors)
    return rows


def constraint_expansion(spec, labels):
    """Sparse Jacobian in the same free-coordinate order as the engine."""
    lookup = {str(level): i for i, level in enumerate(spec["levels"])}
    rows, columns, values = [], [], []
    for j, label in enumerate(labels):
        rows.append(lookup[label["Level"]])
        columns.append(j)
        values.append(1.)
        derived = label.get("DerivedCounterpart")
        if derived:
            rows.append(lookup[derived])
            columns.append(j)
            values.append(-1.)
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(lookup), spec["n_params"]))
