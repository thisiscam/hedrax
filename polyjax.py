from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Tuple, Optional
import math
import numpy as np
import islpy as isl
import jax.numpy as jnp

# -----------------------------
# Your existing table fallback (unchanged)
# -----------------------------


def _fix_params(uset: isl.Set, params: Dict[str, int]) -> isl.Set:
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
    d = p.get_space().dim(isl.dim_type.set)
    return tuple(
        int(p.get_coordinate_val(isl.dim_type.set, t).to_python()) for t in range(d)
    )


def _table_to_indices(domain_str: str, params: Dict[str, int]):
    us = isl.Set(domain_str)
    us_fixed = _fix_params(us, params)

    pts: List[Tuple[int, ...]] = []

    def visit(p):
        pts.append(_point_from_isl_point(p))

    us_fixed.foreach_point(visit)

    if not pts:

        def _unzips_empty(addr):
            arr = np.asarray(addr)
            return np.zeros(arr.shape + (0,), dtype=int)

        return np.zeros((0,), dtype=np.uint8), _unzips_empty

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

    def compress(x: Tuple[int, ...]) -> int:
        return int(sum((x[t] - int(L[t])) * int(S[t]) for t in range(d)))

    addrs = [compress(x) for x in pts]
    idx = np.asarray(addrs, dtype=idx_dtype)

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

    return Solution(idx=idx, unzips=unzips, closed_form=False)


# -----------------------------
# Helpers for reading QPolynomials
# -----------------------------


def _val_to_fraction(v: isl.Val) -> Fraction:
    # isl.Val can be rational; safest is to parse its string form.
    return Fraction(v.to_str())


def _single_set_after_fix(domain_str: str, params: Dict[str, int]) -> isl.Set:
    US = isl.Set(domain_str)
    US = _fix_params(US, params)
    return US


def _dim_bounds_simple_hull(S: isl.Set, t: int) -> Tuple[int, int]:
    Sh = S.simple_hull()
    lo = int(Sh.to_set().dim_min_val(t).to_python())
    hi = int(Sh.to_set().dim_max_val(t).to_python()) + 1  # exclusive
    return lo, hi


def _build_prefix_pwqp(S: isl.Set, t: int, xt_name: str) -> isl.UnionPwQPolynomial:
    """
    Build PwQPolynomial for prefix P_t(..., Xt): count of points with x_t < Xt.
    Earlier set dims (0..t-1) are moved to params, and a fresh param Xt is added by name.
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
        isl.dim_type.set, n_dim, xt_name
    )
    # add the constraint t < Xt
    Spp = Spp.add_constraint(
        isl.Constraint.ineq_from_names(
            Spp.get_space(),
            {
                Spp.get_dim_name(isl.dim_type.set, 0): -1,
                xt_name: 1,
            },
        )
    )
    # move x_t to the last position of params
    Spp = Spp.move_dims(
        isl.dim_type.param, Spp.dim(isl.dim_type.param), isl.dim_type.set, n_dim, 1
    )
    return Spp.card()


@dataclass(frozen=True)
class QuadPrefixRational:
    """P(X) = (A X^2 + B X + C) / D, A,B,C,D ∈ Z, D>0."""

    A: int
    B: int
    C: int
    D: int

    def eval(self, X: int) -> int:
        return (self.A * X * X + self.B * X + self.C) // self.D

    def invert(self, r: int, lo: int, hi: int) -> int:
        A, B, C, D = self.A, self.B, self.C, self.D
        if A == 0:
            # linear fallback
            if B <= 0:
                return lo
            x = (r * D - C) // B
            x = max(lo, min(hi, x))
            while self.eval(x) > r and x > lo:
                x -= 1
            while x + 1 <= hi and self.eval(x + 1) <= r:
                x += 1
            return x
        disc = B * B - 4 * A * (C - r * D)
        if disc < 0:
            return lo
        x0 = int((-B + math.sqrt(float(disc))) // (2 * A))
        x = max(lo, min(hi, x0))
        while self.eval(x) > r and x > lo:
            x -= 1
        while x + 1 <= hi and self.eval(x + 1) <= r:
            x += 1
        return x


@dataclass(frozen=True)
class LinPrefixRational:
    """P(X) = (B X + C) / D."""

    B: int
    C: int
    D: int

    def eval(self, X: int) -> int:
        return (self.B * X + self.C) // self.D

    def invert(self, r: int, lo: int, hi: int) -> int:
        if self.B == 0:
            return lo
        x = (r * self.D - self.C) // self.B
        x = max(lo, min(hi, x))
        while self.eval(x) > r and x > lo:
            x -= 1
        while x + 1 <= hi and self.eval(x + 1) <= r:
            x += 1
        return x


Prefix = Tuple[
    str, object
]  # ("quad", QuadPrefixRational) or ("lin", LinPrefixRational)


@dataclass(frozen=True)
class Solution:
    idx: np.ndarray
    unzips: Callable[[np.ndarray], np.ndarray]
    closed_form: bool


def _extract_prefix_from_qpoly(
    P_pwqp: isl.UnionPwQPolynomial, xt_name: str
) -> Optional[Prefix]:
    """
    Inspect the (single-piece) QPolynomial and extract P(X) = (A X^2 + B X + C)/D in Xt,
    rejecting if:
      - multiple pieces
      - any non-Xt param has positive exponent
      - any 'in' (set) dim has positive exponent
      - any 'div' dim has positive exponent (true quasi-polynomial residue dependence)
      - degree in Xt > 2
    """
    pieces = P_pwqp.get_pieces()
    if len(pieces) != 1:
        return None
    _, qp = pieces[0]  # qp: isl.QPolynomial

    sp = qp.get_space()
    # Identify the Xt parameter index
    xt_pos = sp.find_dim_by_name(isl.dim_type.param, xt_name)
    if xt_pos < 0:
        return None

    A = Fraction(0)
    B = Fraction(0)
    C = Fraction(0)
    n_in = sp.dim(isl.dim_type.in_)  # "in" dims (should be 0 here)
    n_par = sp.dim(isl.dim_type.param)
    n_div = sp.dim(isl.dim_type.div)

    for term in qp.get_terms():
        # reject if any set/in dims present
        for i in range(n_in):
            if term.get_exp(isl.dim_type.in_, i) != 0:
                return None
        # reject if any div dims present (true quasi-poly)
        for i in range(n_div):
            if term.get_exp(isl.dim_type.div, i) != 0:
                return None
        # check params: only Xt may have positive exponent; all others must be 0
        exp_xt = term.get_exp(isl.dim_type.param, xt_pos)
        if exp_xt < 0:
            return None  # shouldn't happen
        for p in range(n_par):
            if p == xt_pos:
                continue
            if term.get_exp(isl.dim_type.param, p) != 0:
                return None  # depends on earlier dims → reject

        # grab coefficient (exact rational)
        coeff = _val_to_fraction(term.get_coefficient_val())

        # degree check + accumulation
        if exp_xt == 0:
            C += coeff
        elif exp_xt == 1:
            B += coeff
        elif exp_xt == 2:
            A += coeff
        else:
            return None  # degree > 2 → reject

    # Classify and normalize to integer numerator / common denominator D
    if A == 0:
        # linear/constant
        Bf, Cf = B, C
        if Bf == 0 and Cf == 0:
            # degenerate zero; treat as linear with B=0 (always 0)
            return ("lin", LinPrefixRational(B=0, C=0, D=1))
        D = math.lcm(Bf.denominator, Cf.denominator)
        Bn = int(Bf * D)
        Cn = int(Cf * D)
        # monotonicity: B >= 0 (prefix must be nondecreasing in X)
        if Bn < 0:
            return None
        return ("lin", LinPrefixRational(B=Bn, C=Cn, D=D))
    else:
        # quadratic
        D = math.lcm(A.denominator, math.lcm(B.denominator, C.denominator))
        An = int(A * D)
        Bn = int(B * D)
        Cn = int(C * D)
        # monotone requirement: An >= 0, and if An==0 then Bn>=0 (already handled)
        if An < 0:
            return None
        return ("quad", QuadPrefixRational(A=An, B=Bn, C=Cn, D=D))


def _synthesize_closed_form_unzips(S: isl.Set) -> Optional[Solution]:
    """
    If every dim yields a single-piece polynomial prefix (degree<=2) in Xt only,
    return (idx=arange(K), unzips(rank->coords)). Otherwise None.
    """
    d = S.dim(isl.dim_type.set)
    per_dim: List[Prefix] = []
    bounds: List[Tuple[int, int]] = []
    for t in range(d):
        xt_name = f"X{t}"
        P_pwqp = _build_prefix_pwqp(S, t, xt_name)
        pref = _extract_prefix_from_qpoly(P_pwqp, xt_name)
        if pref is None:
            return None
        per_dim.append(pref)
        bounds.append(_dim_bounds_simple_hull(S, t))

    # total K and idx = arange(K)
    n_total = S.card().eval_with_dict({})
    idx = np.arange(n_total, dtype=int)

    def unzips(addr):
        a = np.asarray(addr, dtype=np.int64).reshape(-1)
        out = np.zeros((a.shape[0], d), dtype=np.int64)
        for r_i, k in enumerate(a):
            rem = int(k)
            for t in range(d):
                tag, obj = per_dim[t]
                lo, hi = bounds[t]
                x = obj.invert(rem, lo, hi - 1)
                rem -= obj.eval(x)
                out[r_i, t] = x
        return out.reshape(np.asarray(addr).shape + (d,))

    return Solution(idx=idx, unzips=unzips, closed_form=True)


# -----------------------------
# Public hybrid entry
# -----------------------------


def to_indices_hybrid(domain_str: str, params: Dict[str, int]):
    """
    Try closed-form unrank: per-dim prefix is a single-piece polynomial (degree<=2)
    in Xt only (no other params, no divs). If success, returns (idx=arange(K),
    unzips(rank->coords)). Otherwise: fall back to table (idx=addresses, unzips(address->coords)).
    """
    S = _single_set_after_fix(domain_str, params)
    try:
        cf = _synthesize_closed_form_unzips(S)
    except Exception:
        cf = None
    if cf is not None:
        return cf
    return _table_to_indices(domain_str, params)


# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    # 2D triangle
    domain = "[N] -> { [i,j] : 0 <= j < i and 0 <= i < N }"
    sol = to_indices_hybrid(domain, {"N": 10})
    print("triangle: K =", len(sol.idx), "closed_form =", sol.closed_form)
    print("samples:", sol.unzips(sol.idx))

    # 3D triangle x line
    domain3 = "[N] -> { [i,j,k] : 0 <= j < i and 0 <= i < N and 0 <= k < N }"
    sol3 = to_indices_hybrid(domain3, {"N": 10})
    print("tri x line: K =", len(sol3.idx), "closed_form =", sol3.closed_form)
    print("samples:", sol3.unzips(sol3.idx))
