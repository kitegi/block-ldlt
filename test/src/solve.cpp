#include <util.hpp>
#include <doctest.h>
#include <Eigen/Cholesky>
#include <ldlt/solve.hpp>
#include <ldlt/factorize.hpp>
#include <iostream>
#include <util.hpp>

using namespace ldlt;

DOCTEST_TEST_CASE("solve") {
	using T = double;

	for (isize i = 1; i <= 16; ++i) {
		Mat<T, colmajor> a = ldlt_test::rand::positive_definite_rand<T>(i, T(1e2));
		Mat<T, colmajor> l = a;

		Vec<T> b = ldlt_test::rand::vector_rand<T>(i);
		Vec<T> x(i);
		Vec<T> x_eigen(i);
		Vec<long double> x_eigen_upscaled(i);

		auto ldl_view = LdltViewMut<T>{{from_eigen, l}};

		auto x_view = VectorViewMut<T>{from_eigen, x};
		auto b_view = VectorView<T>{from_eigen, b};

		{
			EigenNoAlloc _{};
			factorize(ldl_view);
			solve(x_view, ldl_view.as_const(), b_view);
		}
		x_eigen = a.ldlt().solve(b);
		x_eigen_upscaled =
				a.cast<long double>().ldlt().solve(b.cast<long double>());

		auto err_ours = T((x.cast<long double>() - x_eigen_upscaled).norm());
		auto err_eigen = T((x_eigen.cast<long double>() - x_eigen_upscaled).norm());

		if (err_ours > 0) {
			DOCTEST_CHECK(err_ours / err_eigen < T(1e3));
		}
	}
}
