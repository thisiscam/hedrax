from typing import Dict, List, Tuple, Optional, NamedTuple, Callable, Union

from fractions import Fraction
import math
import numpy as np
import islpy as isl
import jax.numpy as jnp
import jax
from jax import lax


def _bind_parameters(uset: isl.Set, **params: Dict[str, int]) -> isl.Set:
    """
    Fix the parameters of the set to the given values.

    Args:
        uset: The set to fix the parameters of.
        **params: The parameters to fix.

    Returns:
        The set with the parameters fixed.

    Example:
        >>> uset = isl.Set.read_from_str(ctx,
        ...     "[N] -> { [i,j] : 0 <= j < i and 0 <= i < N }"
        ... )
        >>> uset = _bind_parameters(uset, N=10)
        >>> print(uset)
        { [i,j] : 0 <= j < i and 0 <= i < 10 }
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


def _tuple_from_isl_point(p: isl.Point) -> Tuple[int, ...]:
    """
    Convert an isl.Point to a tuple of integers.

    Args:
        p: The isl.Point to convert.

    Returns:
        A tuple of integers representing the point.
    """
    d = p.get_space().dim(isl.dim_type.set)
    return tuple(
        int(p.get_coordinate_val(isl.dim_type.set, t).to_python()) for t in range(d)
    )


class DomainIndexer(NamedTuple):
    """
    A compiled indexer for a domain's lattice points.

    The compiled representation is a tuple of:
    - addresses: a 1D array of integers representing the linearized addresses, or a range object when contiguous (closed-form solution).
    - unravel: a function that maps addresses to multidimensional coordinates.
    """

    addresses: Union[jnp.ndarray, range]
    unravel: Callable[[jnp.ndarray], jnp.ndarray]

    @property
    def is_closed_form(self) -> bool:
        return isinstance(self.addresses, range)


def compile_table_indexer(domain_str: str, **params: Dict[str, int]) -> DomainIndexer:
    """
    Compile a table-based indexer for the domain string.

    Args:
        domain_str: The domain string to convert.
        **params: The parameters to fix.

    Returns:
        A `DomainIndexer` with explicit addresses and an `unravel` function.

    Example:
        >>> domain_str = "[N] -> { [i,j] : 0 <= j < i and 0 <= i < N }"
        >>> sol = compile_table_indexer(domain_str, N=3)
        >>> print(sol.addresses)
        [ 0  1  2  3  5  6  7 10 11 15]
        >>> print(sol.unravel(1))
        [[0 1] [0 2] [0 3] [0 4] [1 2] [1 3] [1 4] [2 3] [2 4] [3 4]]
    """
    us = isl.Set(domain_str)
    us_fixed = _bind_parameters(us, **params)

    pts: List[Tuple[int, ...]] = []

    def visit(p):
        pts.append(_tuple_from_isl_point(p))

    us_fixed.foreach_point(visit)

    if not pts:
        return DomainIndexer(
            addresses=np.zeros((0,), dtype=np.uint8),
            unravel=lambda addr: np.zeros(addr.shape + (0,), dtype=int),
        )

    d = len(pts[0])  # type: ignore
    lower = np.array([min(p[t] for p in pts) for t in range(d)], dtype=int)  # type: ignore
    upper = np.array([max(p[t] for p in pts) + 1 for t in range(d)], dtype=int)  # type: ignore
    size = upper - lower
    stride = np.empty(d, dtype=int)
    acc = 1
    for t in range(d - 1, -1, -1):
        stride[t] = acc
        acc *= int(size[t])

    idx_dtype = np.dtype(int)

    def compress(x: Tuple[int, ...]) -> int:
        return int(sum((x[t] - int(lower[t])) * int(stride[t]) for t in range(d)))

    addrs = [compress(x) for x in pts]
    idx = np.asarray(addrs, dtype=idx_dtype)

    stride_jnp = jnp.asarray(stride, dtype=int)
    size_jnp = jnp.asarray(size, dtype=int)
    lower_jnp = jnp.asarray(lower, dtype=int)

    def unravel(addr: jnp.ndarray) -> jnp.ndarray:
        a = jnp.asarray(addr)
        flat = a.reshape((-1,)).astype(int)
        coords = (flat[:, None] // stride_jnp[None, :]) % size_jnp[None, :] + lower_jnp[
            None, :
        ]
        return coords.reshape((*a.shape, d))

    return DomainIndexer(addresses=idx, unravel=unravel)


# -----------------------------
# Closed-form unrank
# -----------------------------


def _prefix_count_pwqp(S: isl.Set, t: int, new_name: str) -> isl.PwQPolynomial:
    """
    Construct a piecewise quasi-polynomial prefix counter P_t(..., Xt).

    Given a set `S` with set dimensions x_0, ..., x_{d-1}, this returns an
    `isl.PwQPolynomial` `P` such that, for each fixed choice of the earlier
    coordinates x_0, ..., x_{t-1}, the original parameters of `S`, and a new
    parameter `Xt`, `P` evaluates to the number of points in `S` whose t-th
    coordinate is strictly less than `Xt`:

    P(x_0, ..., x_{t-1}, params..., Xt) = | { x in S : x_t < Xt } |.

    To expose the earlier coordinates as functional parameters, the set
    dimensions 0..t-1 of `S` are moved to parameters, and a fresh parameter
    named `new_name` is introduced for `Xt`. The returned polynomial has zero
    set ("in") dimensions.

    Args:
        S: Input `isl.Set` whose lattice points are counted by the prefix.
        t: Zero-based index of the set dimension to prefix on (0 <= t < dim(S)).
        new_name: Name of the fresh parameter representing the bound `Xt`. Must
            be distinct from existing parameter and set-dimension names.

    Returns:
        isl.PwQPolynomial: A piecewise quasi-polynomial in the parameters
        (x_0, ..., x_{t-1}, original params of `S`, `Xt`) giving the count of
        points in `S` with x_t < `Xt`. The result space has zero "in" dims.

    Example:
        >>> s = isl.Set("{ [i,j] : 0 <= j < i and 0 <= i < 10 }")
        >>> _prefix_count_pwqp(s, t=0, new_name="i'")
        [i'] -> { (1/2 * i' + 1/2 * i'^2) : 0 < i' <= 9 }
    """
    Sp = S
    # Move earlier set dims [0..t-1] to parameters so the prefix becomes a
    # function of those dims (now params), the original params, and Xt.
    if t > 0:
        npar = Sp.dim(isl.dim_type.param)
        Sp = Sp.move_dims(isl.dim_type.param, npar, isl.dim_type.set, 0, t)
    Spp = Sp.flat_product(Sp)
    n_dim = Sp.dim(isl.dim_type.set)
    Spp = Spp.remove_dims(isl.dim_type.set, n_dim + 1, n_dim - 1).set_dim_name(
        isl.dim_type.set, n_dim, new_name
    )
    # add the constraint t <= Xt
    Spp = Spp.add_constraint(
        isl.Constraint.ineq_from_names(
            Spp.get_space(),
            {Spp.get_dim_name(isl.dim_type.set, 0): -1, new_name: 1, 1: 0},
        )
    )
    # move x_t to the last position of params
    Spp = Spp.move_dims(
        isl.dim_type.param, Spp.dim(isl.dim_type.param), isl.dim_type.set, n_dim, 1
    )
    return Spp.card()


class Monomial(NamedTuple):
    """
    Integer monomial after clearing denominators: coeff * prod(prev_dims[i] ** exps[i]).

    coeff: integer numerator after multiplying by common denominator D
    exps: tuple of nonnegative integer exponents for earlier dims (length = t)
    """

    coeff: int
    exps: Tuple[int, ...]


class ParametricQuadraticPrefix(NamedTuple):
    """
    Parametric coefficients for P(X) = (A(X_prev) * X^2 + B(X_prev) * X + C(X_prev)) / D

    deg2, deg1, deg0 hold monomials of earlier dims contributing to A, B, C respectively,
    with a shared positive integer denominator D.
    """

    deg2: Tuple[Monomial, ...]
    deg1: Tuple[Monomial, ...]
    deg0: Tuple[Monomial, ...]
    D: int
    t: int


def _extract_parametric_quadratic_prefix(
    P_pwqp: isl.UnionPwQPolynomial, new_name: str
) -> Optional[ParametricQuadraticPrefix]:
    """
    Inspect the (single-piece) QPolynomial and extract
    P(X) = (A(prev)^2 * X^2 + B(prev) * X + C(prev)) / D,
    allowing coefficients to depend on earlier dims (now parameters), but rejecting if:
      - multiple pieces
      - any 'in' (set) dim has positive exponent
      - any 'div' dim has positive exponent (true quasi-polynomial residue dependence)
      - degree in Xt > 2
    Coefficients are represented as lists of integer monomials with a shared denominator D.
    """
    pieces = P_pwqp.get_pieces()
    if len(pieces) != 1:
        return None
    _, qp = pieces[0]  # qp: isl.QPolynomial

    sp = qp.get_space()
    # Identify the Xt parameter index
    xt_pos = sp.find_dim_by_name(isl.dim_type.param, new_name)
    if xt_pos < 0:
        return None

    n_in = sp.dim(isl.dim_type.in_)  # "in" dims (should be 0 here)
    n_par = sp.dim(isl.dim_type.param)
    n_div = sp.dim(isl.dim_type.div)
    # Temporarily accumulate fractional monomials per degree
    deg_to_monos: Dict[int, List[Tuple[Fraction, Tuple[int, ...]]]] = {
        0: [],
        1: [],
        2: [],
    }
    denom_lcm = 1

    for term in qp.get_terms():
        # reject if any set/in dims present
        for i in range(n_in):
            if term.get_exp(isl.dim_type.in_, i) != 0:
                return None
        # reject if any div dims present (true quasi-poly)
        for i in range(n_div):
            if term.get_exp(isl.dim_type.div, i) != 0:
                return None

        # degree in Xt
        exp_xt = term.get_exp(isl.dim_type.param, xt_pos)
        if exp_xt < 0:
            return None
        if exp_xt > 2:
            return None

        # build exponent vector for earlier params (exclude Xt)
        exps_prev: List[int] = []
        for p in range(n_par):
            if p == xt_pos:
                continue
            exps_prev.append(term.get_exp(isl.dim_type.param, p))

        coeff_frac = Fraction(term.get_coefficient_val().to_str())
        denom_lcm = math.lcm(denom_lcm, coeff_frac.denominator)
        deg_to_monos[exp_xt].append((coeff_frac, tuple(exps_prev)))

    # Normalize to integer monomials under common denominator
    def _to_monomials(
        items: List[Tuple[Fraction, Tuple[int, ...]]],
    ) -> Tuple[Monomial, ...]:
        return tuple(
            Monomial(coeff=int(coeff_frac * denom_lcm), exps=exps)
            for coeff_frac, exps in items
        )

    deg2 = _to_monomials(deg_to_monos[2])
    deg1 = _to_monomials(deg_to_monos[1])
    deg0 = _to_monomials(deg_to_monos[0])

    t_inferred = n_par - 1  # number of earlier dims now as params
    return ParametricQuadraticPrefix(
        deg2=tuple(deg2), deg1=tuple(deg1), deg0=tuple(deg0), D=denom_lcm, t=t_inferred
    )


def compile_closed_form_indexer(
    domain_str: str, **params: Dict[str, int]
) -> Optional[DomainIndexer]:
    """
    If every dim yields a single-piece polynomial prefix (degree<=2) in Xt only,
    return (addresses=arange(K), unravel(rank->coords)). Otherwise None.
    """
    S = isl.Set(domain_str)
    S = _bind_parameters(S, **params)
    d = S.dim(isl.dim_type.set)
    per_dim: List[ParametricQuadraticPrefix] = []
    # convert to BasicSet, otherwise reject
    bs = S.get_basic_sets()
    if len(bs) != 1:
        return None
    S = bs[0]
    try:
        S.get_div(0)  # check if any div
        return None
    except Exception:
        pass
    for t in range(d):
        new_name = f"X{t}"
        # reject any constraint has div
        P_pwqp = _prefix_count_pwqp(S, t, new_name)
        pref = _extract_parametric_quadratic_prefix(P_pwqp, new_name)
        if pref is None:
            return None
        per_dim.append(pref)

    # total K and idx = arange(K)
    n_total = S.card().eval_with_dict({})

    def _eval_monomials(monos: Tuple[Monomial, ...], prev: jnp.ndarray) -> jnp.ndarray:
        # prev shape: (t,) possibly empty. Evaluate sum_i coeff_i * prod_j prev[j] ** exps[i, j]
        if len(monos) == 0:
            return jnp.array(0, dtype=int)
        coeffs = jnp.asarray([m.coeff for m in monos], dtype=int)
        exps_mat = jnp.asarray([m.exps for m in monos], dtype=int)

        # If there are no earlier dims, each monomial product is 1
        def _products_no_prev(nrows: int) -> jnp.ndarray:
            return jnp.ones((nrows,), dtype=int)

        def _products_with_prev(
            prev_vec: jnp.ndarray, exps: jnp.ndarray
        ) -> jnp.ndarray:
            bases = prev_vec[None, :]
            powed = jnp.power(bases, exps)
            return jnp.prod(powed, axis=1)

        products = jax.lax.cond(
            prev.size == 0,
            lambda _: _products_no_prev(exps_mat.shape[0]),
            lambda _: _products_with_prev(prev, exps_mat),
            operand=None,
        )
        return jnp.sum(coeffs * products, dtype=int)

    def _eval_coeffs(
        prefix: ParametricQuadraticPrefix, prev: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        A_num = _eval_monomials(prefix.deg2, prev)
        B_num = _eval_monomials(prefix.deg1, prev)
        C_num = _eval_monomials(prefix.deg0, prev)
        return A_num, B_num, C_num, prefix.D

    def _eval_prefix_count(
        prefix: ParametricQuadraticPrefix, prev: jnp.ndarray, X: jnp.ndarray
    ) -> jnp.ndarray:
        X = jnp.asarray(X)
        A_num, B_num, C_num, D = _eval_coeffs(prefix, prev)
        return jnp.floor_divide(A_num * X * X + B_num * X + C_num, D)

    def _invert_prefix_count(
        prefix: ParametricQuadraticPrefix, prev: jnp.ndarray, r: jnp.ndarray
    ) -> jnp.ndarray:
        # Closed-form inversion using sqrt for quadratic and direct formula for linear.
        r_arr = jnp.asarray(r)

        A_num, B_num, C_num, D = _eval_coeffs(prefix, prev)
        is_const = (A_num == 0) & (B_num == 0)
        # Handle linear vs quadratic using cond to keep JAX-friendly control flow
        is_linear = A_num == 0

        def _compute_x_lin(_: None) -> jnp.ndarray:
            # x = floor((D*r - C)/B)
            return jnp.floor_divide(D * r_arr - C_num, B_num).astype(int)

        def _compute_x_quad(_: None) -> jnp.ndarray:
            # x = floor(( -B + sqrt(B^2 - 4*A*(C - D*r)) ) / (2*A))
            disc = B_num * B_num - 4 * A_num * (C_num - D * r_arr)
            root = jnp.sqrt(disc)
            denom = 2.0 * A_num
            return jnp.floor_divide(-B_num + root, denom).astype(int)

        x_nonconst = jax.lax.cond(
            is_linear,
            _compute_x_lin,
            _compute_x_quad,
            operand=None,
        )

        x_base = jax.lax.cond(
            is_const,
            lambda _: jnp.zeros_like(r_arr, dtype=int),
            lambda _: x_nonconst,
            operand=None,
        )
        x0 = x_base + 1
        return x0.astype(int)

    def _unravel_one(addr_scalar: jnp.ndarray) -> jnp.ndarray:
        rem = jnp.asarray(addr_scalar)
        xs = []
        # Sequentially recover each coordinate; d is a Python int, so this loop is static.
        for t in range(d):
            obj = per_dim[t]
            prev = (
                jnp.asarray(xs, dtype=int)
                if len(xs) > 0
                else jnp.asarray([], dtype=int)
            )
            x_t = _invert_prefix_count(obj, prev, rem)
            rem = rem - _eval_prefix_count(obj, prev, x_t - 1)
            xs.append(x_t)
        return jnp.stack(xs, axis=0).astype(int)

    def unravel(addr):
        a = jnp.asarray(addr)
        flat = a.reshape((-1,))
        coords = jax.vmap(_unravel_one)(flat)
        return coords.reshape((*a.shape, d)).astype(int)

    return DomainIndexer(addresses=range(n_total), unravel=unravel)


# -----------------------------
# Public hybrid entry
# -----------------------------


def compile_indexer(domain_str: str, **params: Dict[str, int]):
    """
    Try closed-form unrank: per-dim prefix is a single-piece polynomial (degree<=2)
    in Xt only (no other params, no divs). If success, returns (addresses=arange(K),
    unravel(rank->coords)). Otherwise: fall back to table (addresses=array, unravel(address->coords)).
    """
    cf = compile_closed_form_indexer(domain_str, **params)
    if cf is not None:
        return cf
    return compile_table_indexer(domain_str, **params)


def foreach(constr: str, **params: Dict[str, int]):
    """
    A decorator that applies a function to each element in a domain.

    Args:
        constr: The constraint string of the domain. The variables must be the arguments of the wrapped function.
        **params: The parameters to fix, same as `compile_table_indexer`.

    Returns:
        Use as a decorator to a function. The result of the function applied to each element in the domain.

    Example:
        >>> N = 10
        >>> r = jax.array_ref(jnp.zeros(N))
        >>> @foreach(f"0 <= j < i < {N}")
        ... def _(i, j):
        ...   r[i] += jnp.sin(i) * jnp.cos(j)**2
        >>> print(r)
        [ 0.          0.00000000  0.00000000  0.00000000  0.00000000  0.00000000
          0.00000000  0.00000000  0.00000000  0.00000000]
    """

    def decorator(body: Callable[[int, ...], None]):
        coordinate_names = [
            name
            for name in body.__code__.co_varnames
            if name not in ["self", "args", "kwargs"]
        ]
        # format the constr into a domain string
        domain_str = f"{{ [{', '.join(coordinate_names)}] : {constr} }}"
        addresses, unravel = compile_table_indexer(domain_str, **params)
        _, res = lax.scan(lambda _, elts: (None, body(*unravel(elts))), None, addresses)
        return res

    return decorator


# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    # 2D triangle
    domain = "[N] -> { [i, j] : i + 10 <= j < 2 * i + 10 and 1 <= i < N + 1}"
    sol = compile_indexer(domain, N=10)
    print("triangle: K =", len(sol.addresses), "closed_form =", sol.is_closed_form)
    print(sol.unravel(35))
    print("samples:", sol.unravel(sol.addresses))

    sol1 = compile_table_indexer(domain, N=10)
    print("triangle: K =", len(sol1.addresses), "closed_form =", sol1.is_closed_form)
    print(sol1.unravel(35))
    print("samples:", sol1.unravel(sol1.addresses))

    N = 10
    r = jax.array_ref(jnp.zeros(N))

    @foreach(f"0 <= j < i < {N}")
    def _(i, j):
        r[i] += jnp.sin(i) * jnp.cos(j) ** 2

    rp = jax.array_ref(jnp.zeros(N))
    for i in range(N):
        for j in range(i):
            rp[i] += jnp.sin(i) * jnp.cos(j) ** 2

    print(r)
    print(rp)
