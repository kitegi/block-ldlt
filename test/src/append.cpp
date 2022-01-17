#include <Eigen/Core>
#include <doctest.h>

#include <iostream>
#include <util.hpp>

#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>

using Scalar = double;
using namespace ldlt;

DOCTEST_TEST_CASE("row add") {
	isize n = 7;
	using T = f64;

	using Mat = ::Mat<T, colmajor>;
	using Vec = ::Vec<T>;

	Mat m = ldlt_test::rand::positive_definite_rand(n + 1, T(1e2));

	Mat l_target(n + 1, n + 1);
	Mat l_in(n, n);
	Mat l_out(n + 1, n + 1);
	l_out.setZero();

	using LdltView = ldlt::LdltView<T>;
	using LdltViewMut = ldlt::LdltViewMut<T>;
	using MatrixView = ldlt::MatrixView<T, colmajor>;
	using MatrixViewMut = ldlt::MatrixViewMut<T, colmajor>;

	bool bool_values[] = {false, true};
	for (bool inplace : bool_values) {
		// factorize input matrix
		l_in = m.topLeftCorner(n, n);
		ldlt::factorize(LdltViewMut{{from_eigen, l_in}});

		if (inplace) {
			l_out.topLeftCorner(n, n) = l_in.topLeftCorner(n, n);
		}

		// append row
		ldlt::row_append(
				LdltViewMut{
						MatrixViewMut{from_eigen, l_out},
				},
				inplace ? (LdltView{MatrixView{from_eigen, l_out}.block(0, 0, n, n)})
								: (LdltView{{from_eigen, l_in}}),
				{from_eigen, Vec(m.row(n))});

		// compute target
		l_target = m;
		ldlt::factorize(LdltViewMut{{from_eigen, l_target}});

		auto eps = Scalar(1e-10);
		DOCTEST_CHECK(
				Mat((l_target - l_out).triangularView<Eigen::Lower>()).norm() <= eps);
	}
}
