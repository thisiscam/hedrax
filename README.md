# HedraX: Polyhedral Indexer for Static-Shape Programming in JAX

HedraX is a polyhedral indexer for static-shape programming in JAX.
It is designed to help you write static-shape programs in JAX by providing a way to index into statically-shaped, bounded polyhedral domains.

## Installation

For now, you can install it from the repository:

```bash
$ pip install git+https://github.com/thisiscam/hedrax.git
```

## Background

In JAX, we must write static-shape programs.
This means that we must know the shape of the arrays we are working with at compile time.

In some cases, certain shapes are tedious to express in terms of JAX's static shape system.
For example, consider the following triangular domain:

```python
for i in range(N):
    for j in range(i):
        A[i, j] = f(i, j)
```

The above is expressible in JAX, but it is tedious to write:

```python
# Map k ∈ [0, N*(N+1)//2) to (i, j) with 0 ≤ j ≤ i
def tri_index(k):
    k = jnp.asarray(k)
    i = jnp.floor((jnp.sqrt(8.0 * k + 1.0) - 1.0) / 2.0).astype(jnp.int32)
    ti = (i * (i + 1)) // 2                  # triangular number T_i
    j = (k - ti).astype(jnp.int32)
    return i, j

def fill_lower_tri(N, f, dtype=jnp.float32):
    M = N * (N + 1) // 2
    A0 = jnp.zeros((N, N), dtype=dtype)

    def body(A, k):
        i, j = tri_index(k)
        A = lambda a: a.at[i, j].set(f(i, j))
        return A, None

    A, _ = lax.scan(body, A0, jnp.arange(M, dtype=jnp.int32))
    return A
```

What happened here is that we packed the triangular domain into a continuous range of integers `range(N * (N + 1) // 2)`,
and we scan over it.
Then, inside the scan body, we unpack the encoded index `k` into `(i, j)` by solving quadratic equations.

Another way to express this is to use a table-based indexer, where we first build a table of indices,
then we extract (i, j) from each index:

Both are insanely tedious to write, and it is easy to make mistakes.

HedraX provides a way to express this programming pattern in a more concise way.

## Usage

You simply write

```python
import hedrax as hdx

domain = "[N] -> { [i, j] : 0 <= j < i and 0 <= i < N }"
addresses, unravel = hdx.compile_indexer(domain, N=10)

def body(A, k):
    i, j = unravel(k)
    A = A.at[i, j].set(f(i, j))
    return A, None

A, _ = jax.scan(body, A0, addresses)
```

That's it!

## Implementation

HedraX is implemented using the [isl](https://libisl.sourceforge.io/) library, a library for manipulating Presburger arithmetic.
There are two main indexers:

1. A table-based indexer based on enumerating the lattice points of the domain, then building a lookup table.
2. A closed-form indexer by using Barvinok's algorithm to find inverse functions for the domain.

The table-based indexer should work for ANY polyhedral domain --- including non-convex domains by taking unions of convex domains ---
as long as it is finite and reasonly sized within the range of 32/64-bit integers, depending on what your JAX dtype is.

The closed-form indexer works for domains that are convex and contiguous (i.e. the domain does not have strides),
and where the inverse function for recovering each dimension is simpler than solving a quasi-quadratic equation.
