#include <Eigen/Core>
#include <doctest.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <util.hpp>

#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>

using namespace ldlt;
using T = f64;

DOCTEST_TEST_CASE("row delete") {
	isize n = 7;

	using Mat = ::Mat<T, colmajor>;

	Mat m = ldlt_test::rand::positive_definite_rand(n, T(1e2));
	m.setRandom();
	m = m * m.transpose();

	Mat l_target(n - 1, n - 1);

	Mat l_in(n, n);

	Mat l_out_storage(n, n);
	l_out_storage.setZero();

	auto l_out = l_out_storage.topLeftCorner(n - 1, n - 1);

	using LdltViewMut = ldlt::LdltViewMut<T>;

	for (isize idx = 0; idx < n; ++idx) {
		// factorize input matrix
		l_in = m;
		ldlt::factorize(LdltViewMut{{from_eigen, l_in}});

		l_out_storage = l_in;

		// delete ith row
		ldlt::row_delete(LdltViewMut{{from_eigen, l_out_storage}}, idx);

		// compute target
		{
			// delete idx'th row and column
			isize rem = n - idx - 1;
			l_target.topLeftCorner(idx, idx) = m.topLeftCorner(idx, idx);
			l_target.bottomLeftCorner(rem, idx) = m.bottomLeftCorner(rem, idx);

			l_target.topRightCorner(idx, rem) = m.topRightCorner(idx, rem);
			l_target.bottomRightCorner(rem, rem) = m.bottomRightCorner(rem, rem);

			ldlt::factorize(LdltViewMut{{from_eigen, l_target}});
		}

		T eps = T(1e-10);
		DOCTEST_CHECK(
				Mat((l_target - l_out).triangularView<Eigen::Lower>()).norm() <= eps);
	}
}
