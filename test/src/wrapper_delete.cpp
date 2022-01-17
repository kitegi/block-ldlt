#include <ldlt/ldlt.hpp>
#include <util.hpp>
#include <doctest.h>

using namespace ldlt;

using T = f64;
TEST_CASE("delete") {
	T const eps = std::numeric_limits<T>::epsilon() * T(1e3);

	for (isize n = 2; n < 32; ++n) {
		auto const mat = ldlt_test::rand::positive_definite_rand<T>(n, 1e2);

		for (isize i = 0; i < n; ++i) {
			auto const mat_reduced = [&] {
				auto mat_reduced = mat;
				isize rem = n - i - 1;
				mat_reduced.middleCols(i, rem) =
						mat_reduced.middleCols(i + 1, rem).eval();
				mat_reduced.conservativeResize(n, n - 1);
				mat_reduced.middleRows(i, rem) =
						mat_reduced.middleRows(i + 1, rem).eval();
				mat_reduced.conservativeResize(n - 1, n - 1);
				return mat_reduced;
			}();

			auto ldl = Ldlt<T>{};
      LDLT_MAKE_STACK(stack, Ldlt<T>::factor_req(n));
      ldl.factor(mat, LDLT_FWD(stack));
			ldl.delete_at(i);

			CHECK((mat_reduced - ldl.dbg_reconstructed_matrix()).norm() < eps);
		}
	}
}
