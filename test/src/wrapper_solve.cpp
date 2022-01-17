#include <ldlt/ldlt.hpp>
#include <util.hpp>
#include <doctest.h>
#include <fmt/ostream.h>

using namespace ldlt;
DOCTEST_TEST_CASE("solve apply") {
	using T = f64;
	isize n = 13;
	using Mat = ::Mat<T, colmajor>;
	auto in = ldlt_test::rand::positive_definite_rand<T>(n, 1e2);
	auto rhs = ldlt_test::rand::vector_rand<T>(n);

	Ldlt<T> ldl;
	auto sol = rhs;

	LDLT_MAKE_STACK(
			stack, (Ldlt<T>::factor_req(n) | Ldlt<T>::solve_in_place_req(n)));
	ldl.factor(in, LDLT_FWD(stack));
	ldl.solve_in_place(sol, LDLT_FWD(stack));

	DOCTEST_CHECK((in * sol - rhs).norm() <= 1e-3);

	auto l = ldl.l();
	auto lt = ldl.lt();
	auto d = ldl.d().asDiagonal();
	auto p = ldl.p();
	auto pt = ldl.pt();
	DOCTEST_CHECK((p * Mat(l) * d * Mat(lt) * pt - in).norm() <= 1e-3);
}

DOCTEST_TEST_CASE("rank update solve") {
	using T = f64;
	isize n = 13;
	auto in = ldlt_test::rand::positive_definite_rand<T>(n, 1e2);
	auto z = ldlt_test::rand::vector_rand<T>(n);

	auto rhs = ldlt_test::rand::vector_rand<T>(n);

	Ldlt<T> ldl;
	auto sol = rhs;

	LDLT_MAKE_STACK(
			stack,
			(Ldlt<T>::factor_req(n) | Ldlt<T>::solve_in_place_req(n) |
	     Ldlt<T>::rank_one_update_req(n)));

	ldl.factor(in, LDLT_FWD(stack));

	{
		in += 2.5 * z * z.transpose();
		ldl.rank_one_update(z, 2.5, LDLT_FWD(stack));
	}

	ldl.solve_in_place(sol, LDLT_FWD(stack));

	DOCTEST_CHECK((in * sol - rhs).norm() <= 1e-3);
	DOCTEST_CHECK((ldl.dbg_reconstructed_matrix() - in).norm() <= 1e-3);
}
