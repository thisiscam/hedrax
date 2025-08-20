"""Minimal stateless unranking for Presburger (ISL) domains.

This module provides utilities to enumerate integer points of an ISL domain
at fixed parameter values and to construct a pair `(unzips, idx)` where
`unzips` maps a compressed address back to coordinates and `idx` maps
rank to the compressed address.

Notes:
  - Requires `islpy`.
  - When `dtype` is set to "auto", a minimal unsigned integer type is chosen
    to fit the address space of the parameter-specific bounding box inferred
    from the enumerated points.
"""

from typing import Dict, List, Tuple
import numpy as np
import islpy as isl
import jax.numpy as jnp


def _fix_params(uset: isl.UnionSet, params: Dict[str, int]) -> isl.UnionSet:
    """Fix parameter dimensions of an ISL set/union set to concrete values.

    Args:
      uset: ISL set or union set whose symbolic parameters will be fixed.
      params: Mapping from parameter name to integer value.

    Returns:
      An `isl.UnionSet` with all parameters fixed and projected out of the
      space.

    Raises:
      KeyError: If a parameter name in `params` is not found in the domain
        space.
    """
    ctx = uset.get_ctx()
    sp = uset.get_space()
    for name, value in params.items():
        pos = sp.find_dim_by_name(isl.dim_type.param, name)
        if pos < 0:
            raise KeyError(f"Parameter '{name}' not found in domain space.")
        uset = uset.fix_val(
            isl.dim_type.param, pos, isl.Val.int_from_si(ctx, int(value))
        )
    return uset.project_out_all_params()


def _point_from_isl_point(p: isl.Point) -> Tuple[int, ...]:
    """Convert an ISL point to a Python tuple of integer coordinates.

    Args:
      p: ISL point in a set space.

    Returns:
      A tuple of integers, one per set dimension.
    """
    d = p.get_space().dim(isl.dim_type.set)
    return tuple(
        int(p.get_coordinate_val(isl.dim_type.set, t).to_python()) for t in range(d)
    )


def to_indices(domain_str: str, params: Dict[str, int]):
    """Compute address mapping and inverse for an ISL domain at fixed params.

    This enumerates all integer points in the domain (order unspecified),
    optionally deduplicates across union pieces, infers a parameter-specific
    bounding box, and returns a vectorized inverse mapping along with the
    compressed addresses for each enumerated point.

    Args:
      domain_str: ISL set/union set string, possibly with parameters
        (for example, "[N] -> { [i, j] : 0 <= j < i and 0 <= i < N }").
      params: Mapping from parameter name to concrete integer value.

    Returns:
      A pair `(unzips, idx)` where:
        - `unzips`: Callable `unzips(addr) -> coords` that supports scalars and
          arrays and returns coordinates with shape `(..., D)`.
        - `idx`: 1D NumPy array of length K mapping rank k to compressed address.

    Raises:
      KeyError: If a parameter in `params` is not found in the domain space.

    Examples:
      >>> domain = "[N] -> { [i, j] : 0 <= j < i and 0 <= i < N }"
      >>> idx, unzips = to_indices(domain, {"N": 4})
      >>> unzips(idx).shape  # (K, 2) where K is the number of points
      (4, 2)
    """
    us = isl.Set(domain_str)
    us_fixed = _fix_params(us, params)  # ensure bounded

    # 1) Enumerate points using foreach_point over each set in the union
    pts: List[Tuple[int, ...]] = []

    def visit(p):
        pts.append(_point_from_isl_point(p))

    us_fixed.foreach_point(visit)

    if not pts:
        # Empty domain
        def _unzips_empty(addr):
            arr = np.asarray(addr)
            return np.zeros(arr.shape + (0,), dtype=int)

        return _unzips_empty, np.zeros((0,), dtype=np.uint8)

    # 2) Infer param-specific box bounds L/U and strides S
    d = len(pts[0])
    L = np.array([min(p[t] for p in pts) for t in range(d)], dtype=int)
    U = np.array([max(p[t] for p in pts) + 1 for t in range(d)], dtype=int)
    M = U - L
    S = np.empty(d, dtype=int)
    acc = 1
    for t in range(d - 1, -1, -1):
        S[t] = acc
        acc *= int(M[t])

    idx_dtype = np.dtype(int)

    # 4) Build idx by compressing each point once
    def compress(x: Tuple[int, ...]) -> int:
        return int(sum((x[t] - int(L[t])) * int(S[t]) for t in range(d)))

    addrs = [compress(x) for x in pts]
    idx = np.asarray(addrs, dtype=idx_dtype)

    # 5) Build unzips closure (inverse of compress for these params)
    S_jnp = jnp.asarray(S, dtype=int)
    M_jnp = jnp.asarray(M, dtype=int)
    L_jnp = jnp.asarray(L, dtype=int)

    def unzips(addr: jnp.ndarray) -> jnp.ndarray:
        a = jnp.asarray(addr)
        flat = a.reshape((-1,))
        flat_i64 = flat.astype(int)
        coords = ((flat_i64[:, None] // S_jnp[None, :]) % M_jnp[None, :]) + L_jnp[
            None, :
        ]
        return coords.reshape((*a.shape, d))

    return idx, unzips


if __name__ == "__main__":
    domain = (
        "[N] -> { [i,j] : 0 <= j < i and 0 <= i < N or (N <= i < N * 2 and j = 4) }"
    )
    idx, unzips = to_indices(domain, {"N": 100})
    print("K =", idx)
    print(unzips(idx))
