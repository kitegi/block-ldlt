#include <Eigen/Cholesky>
#include <iostream>
#include <ldlt/factorize.hpp>
#include <util.hpp>
#include <limits>
#include <doctest.h>
#include <ldlt/ldlt.hpp>

using namespace ldlt;

template <typename T, Layout L>
struct Data {
	Mat<T, L> mat;
	Mat<T, colmajor> ld;
};

template <typename T, Layout L>
auto generate_data(isize n) -> Data<T, L> {
	ldlt_test::rand::set_seed(u64(n));
	Mat<T, L> mat = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));
	Mat<T, colmajor> l(n, n);
	return {
			VEG_FWD(mat),
			VEG_FWD(l),
	};
}

template <typename T, Layout L, typename S>
auto ldlt_roundtrip_error(Data<T, L>& data, S strategy) -> T {
	auto const& mat = data.mat;
	auto& ld = data.ld;
	ld = mat;

	auto ldl_view = LdltViewMut<T>{{from_eigen, ld}};

	{
		EigenNoAlloc _{};
		factorize(ldl_view, strategy);
	}

	return (reconstruct(ldl_view.as_const()) - mat).norm();
}

template <typename T, Layout L, typename S>
auto roundtrip_test(isize n, S strategy) -> T {
	auto data = generate_data<T, L>(n);
	return ::ldlt_roundtrip_error(data, strategy) /
	       std::numeric_limits<T>::epsilon();
}

DOCTEST_TEST_CASE("factorize: roundtrip") {
	isize min = 1;
	isize max = 64;
	using Scalar = f32;
	constexpr auto L = colmajor;

	isize block_sizes[] = {1, 2, 4, 16, 64};

	for (auto bs : block_sizes) {
		auto tag = factorization_strategy::blocked(bs);
		for (isize i = min; i <= max; ++i) {
			DOCTEST_CHECK(roundtrip_test<Scalar, L>(i, tag) <= Scalar(1e3));
		}
		DOCTEST_CHECK(roundtrip_test<Scalar, L>(200, tag) <= Scalar(1e3));
		DOCTEST_CHECK(roundtrip_test<Scalar, L>(256, tag) <= Scalar(1e3));
	}
}
