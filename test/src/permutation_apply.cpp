#include <ldlt/factorize.hpp>
#include <util.hpp>
#include <doctest.h>

using namespace ldlt;
DOCTEST_TEST_CASE("permute apply") {
	using T = f32;
	isize n = 13;
	auto in = ldlt_test::rand::matrix_rand<T>(n, n);
	in = in + in.transpose().eval();
	auto perm = std::vector<isize>(usize(n));
	auto perm_inv = std::vector<isize>(usize(n));
	ldlt::detail::compute_permutation<T>( //
			perm.data(),
			perm_inv.data(),
			{from_eigen, in.diagonal()});

	i32 syms[] = {0, -1, 1};

	LDLT_WORKSPACE_MEMORY(work, Uninit, Mat(n, n), LDLT_CACHELINE_BYTES, T);
	for (auto sym : syms) {
		auto out = in;

		ldlt::detail::apply_permutation_sym_work<T>( //
				{from_eigen, out},
				perm.data(),
				work,
				sym);

		switch (sym) {
		case 0: {
			for (isize i = 0; i < n; ++i) {
				for (isize j = 0; j < n; ++j) {
					DOCTEST_CHECK(out(i, j) == in(perm[usize(i)], perm[usize(j)]));
				}
			}
			break;
		}
		case -1: {
			// lower triangular, col <= row
			for (isize i = 0; i < n; ++i) {
				for (isize j = 0; j <= i; ++j) {
					DOCTEST_CHECK(out(i, j) == in(perm[usize(i)], perm[usize(j)]));
				}
			}
			break;
		}
		case 1: {
			// upper triangular, col >= row
			for (isize i = 0; i < n; ++i) {
				for (isize j = i; j < n; ++j) {
					DOCTEST_CHECK(out(i, j) == in(perm[usize(i)], perm[usize(j)]));
				}
			}
			break;
		}
		default:
			break;
		}
	}
}
