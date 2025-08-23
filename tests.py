import numpy as np
import pytest
import islpy as isl

from hedrax import (
    compile_indexer,
    compile_table_indexer,
    compile_closed_form_indexer,
    DomainIndexer,
    _prefix_count_pwqp,
)


def test_prefix_count_pwqp_eval_example():
    s = isl.Set("{ [i,j] : 0 <= j < i and 0 <= i < 10 }")
    P = _prefix_count_pwqp(s, t=0, new_name="X")
    assert P.eval_with_dict({"N": 10, "X": 3}) == 6


def _enumerate_triangle_pairs(N: int):
    return [(j, i) for i in range(N) for j in range(i)]


@pytest.mark.parametrize(
    "N,domain",
    [
        (3, "[N] -> { [j,i] : 0 <= j < i and 0 <= i < N }"),
        (5, "[N] -> { [j,i] : 0 <= j < i and 0 <= i < N }"),
        (10, "[N] -> { [j,i] : 0 <= j < i and 0 <= i < N }"),
    ],
)
def test_compile_indexer_triangle_closed_form(N: int, domain: str):
    sol: DomainIndexer = compile_indexer(domain, N=N)

    expected_K = N * (N - 1) // 2
    assert isinstance(sol, DomainIndexer)

    assert sol.is_closed_form is True
    assert len(sol.addresses) == expected_K

    # Unravel all addresses and validate constraints and count
    coords = np.asarray(sol.unravel(list(sol.addresses)))
    assert coords.shape == (expected_K, 2)
    # All satisfy 0 <= j < i < N
    assert np.all(coords[:, 0] >= 0)
    assert np.all(coords[:, 1] >= 0)
    assert np.all(coords[:, 0] < coords[:, 1])
    assert np.all(coords[:, 1] < N)

    # Set equality with the simple enumeration
    expected = set(_enumerate_triangle_pairs(N))
    got = set(map(tuple, coords.tolist()))
    assert got == expected


def test_compile_table_indexer_triangle_small():
    N = 5
    domain = "[N] -> { [j,i] : 0 <= j < i and 0 <= i < N }"
    sol: DomainIndexer = compile_table_indexer(domain, N=N)

    expected_K = N * (N - 1) // 2
    assert isinstance(sol, DomainIndexer)
    assert sol.is_closed_form is False
    assert isinstance(sol.addresses, np.ndarray)
    assert sol.addresses.shape == (expected_K,)

    coords = np.asarray(sol.unravel(sol.addresses))
    assert coords.shape == (expected_K, 2)
    assert np.all(coords[:, 0] >= 0)
    assert np.all(coords[:, 1] >= 0)
    assert np.all(coords[:, 0] < coords[:, 1])
    assert np.all(coords[:, 1] < N)

    expected = set(_enumerate_triangle_pairs(N))
    got = set(map(tuple, coords.tolist()))
    assert got == expected


def test_compile_indexer_triangle_times_line_closed_form():
    N = 10
    domain3 = "[N] -> { [i,j,k] : 0 <= j < i and 0 <= i < N and 0 <= k < N }"
    sol3: DomainIndexer = compile_indexer(domain3, N=N)

    expected_K = (N * (N - 1) // 2) * N
    assert isinstance(sol3, DomainIndexer)
    assert sol3.is_closed_form is True
    assert len(sol3.addresses) == expected_K

    coords3 = np.asarray(sol3.unravel(list(sol3.addresses)))
    assert coords3.shape == (expected_K, 3)
    j = coords3[:, 1]
    i = coords3[:, 0]
    k = coords3[:, 2]
    assert np.all(j < i)
    assert np.all(i < N)
    assert np.all(k >= 0)
    assert np.all(k < N)


@pytest.mark.parametrize(
    "N,domain",
    [
        # (1, "[N] -> { [i] : 0 <= i < N }"),
        (3, "[N] -> { [i] : 0 <= i < N }"),
        (7, "[N] -> { [i] : 0 <= i < N }"),
        (2, "[N] -> { [i,j] : 0 <= i < N and 0 <= j < N }"),
        (5, "[N] -> { [i,j] : 0 <= i < N and 0 <= j < N }"),
        # (1, "[N] -> { [i] : exists a : i = 2a and 0 <= i < N }"),
        (5, "[N] -> { [i] : exists a : i = 2a and 0 <= i < N }"),
        (10, "[N] -> { [i] : exists a : i = 2a and 0 <= i < N }"),
    ],
)
def test_table_contiguity_controls_closed_form(N: int, domain: str):
    # Table indexer
    sol_table: DomainIndexer = compile_table_indexer(domain, N=N)
    idx = np.asarray(sol_table.addresses)
    # contiguity: sorted indices form an arithmetic progression with step=1
    if idx.size == 0:
        is_contig = True
    else:
        i_min = int(idx.min())
        i_max = int(idx.max())
        is_contig = (i_max - i_min + 1 == idx.size) and (
            np.unique(idx).size == idx.size
        )

    # Closed-form indexer
    sol_cf = compile_closed_form_indexer(domain, N=N)
    has_cf = sol_cf is not None

    # Hybrid should choose closed form iff available
    sol_hybrid: DomainIndexer = compile_indexer(domain, N=N)
    assert sol_hybrid.is_closed_form == has_cf

    # Expectation: closed-form is available iff table addresses are contiguous
    assert not is_contig or has_cf  # is_contig implies has_cf

    # Validate that hybrid and table enumerate the same set of points
    coords_table = np.asarray(sol_table.unravel(sol_table.addresses))
    coords_hybrid = np.asarray(sol_hybrid.unravel(list(sol_hybrid.addresses)))
    set_table = set(map(tuple, coords_table.tolist()))
    set_hybrid = set(map(tuple, coords_hybrid.tolist()))
    assert set_table == set_hybrid
