#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <util.hpp>

#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>
#include <veg/util/dbg.hpp>

#define LDLT_BENCHMARK_MAIN BENCHMARK_MAIN    /* NOLINT */
#define LDLT_BENCHMARK BENCHMARK              /* NOLINT */
#define LDLT_BENCHMARK_TPL BENCHMARK_TEMPLATE /* NOLINT */

using namespace ldlt;

using T = f64;

void bench_ours___ /* NOLINT */ (benchmark::State& s) {
	auto n = isize(s.range(0));

	ldlt_test::rand::set_seed(0);
	auto a = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));
	auto ldl = LdltViewMut<T>{{from_eigen, a.topLeftCorner(n, n)}};
	factorize(ldl);

	Vec<T> z(n);
	z.setConstant(-0.0);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(z.data());

	for (auto _ : s) {
		rank1_update(ldl, {from_eigen, z}, 1.0);
		benchmark::ClobberMemory();
	}
}

// void bench_ours__r /* NOLINT */ (benchmark::State& s) {
// 	auto n = isize(s.range(0));
// 	auto r = isize(s.range(1));

// 	ldlt_test::rand::set_seed(0);
// 	auto a = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));
// 	auto ldl = LdltViewMut<T>{{from_eigen, a}};
// 	factorize(ldl);

// 	Mat<T, colmajor> z(n, r);
// 	z.setConstant(-0.0);
// 	Vec<T> alpha = ldlt_test::rand::vector_rand<T>(r);

// 	benchmark::DoNotOptimize(a.data());
// 	benchmark::DoNotOptimize(z.data());

// 	for (auto _ : s) {
// 		detail::rank_r_update(
// 				ldl, ldl.as_const(), {from_eigen, z}, {from_eigen, alpha});
// 		benchmark::ClobberMemory();
// 	}
// }
// void bench_ours_r4(benchmark::State& s) {
// 	VEG_ASSERT(1 == 2);
// 	auto n = isize(s.range(0));
// 	auto r = isize(s.range(1));

// 	ldlt_test::rand::set_seed(0);
// 	auto a = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));
// 	auto ldl = LdltViewMut<T>{{from_eigen, a}};
// 	factorize(ldl);

// 	Mat<T, colmajor> z(n, r);
// 	z.setConstant(-0.0);
// 	Vec<T> alpha = ldlt_test::rand::vector_rand<T>(r);

// 	benchmark::DoNotOptimize(a.data());
// 	benchmark::DoNotOptimize(z.data());

// 	for (auto _ : s) {

// 		isize i = 0;
// 		while (true) {
// 			if (i == r) {
// 				break;
// 			}

// 			isize r_block = ldlt::detail::min2(r - i, isize(4));
// 			detail::rank_r_update(
// 					ldl,
// 					ldl.as_const(),
// 					{from_eigen, z.middleCols(i, r_block)},
// 					{from_eigen, alpha.segment(i, r_block)});
// 			i += r_block;
// 		}
// 		benchmark::ClobberMemory();
// 	}
// }

void bench_eigen__ /* NOLINT */ (benchmark::State& s) {
	auto n = isize(s.range(0));

	ldlt_test::rand::set_seed(0);
	auto a = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));

	auto ldl = a.ldlt();

	Vec<T> z(n);
	z.setConstant(-0.0);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(const_cast<T*>(ldl.matrixLDLT().data()));

	for (auto _ : s) {
		ldl.rankUpdate(z, 1.0);
		benchmark::ClobberMemory();
	}
}
void bench_eigen_r(benchmark::State& s) {
	auto n = isize(s.range(0));
	auto r = isize(s.range(1));

	ldlt_test::rand::set_seed(0);
	auto a = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));

	auto ldl = a.ldlt();

	Mat<T, colmajor> z(n, r);
	z.setConstant(-0.0);
	Vec<T> alpha = ldlt_test::rand::vector_rand<T>(r);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(const_cast<T*>(ldl.matrixLDLT().data()));

	for (auto _ : s) {
		for (isize i = 0; i < r; ++i) {
			ldl.rankUpdate(z.col(i), alpha(i));
		}
		benchmark::ClobberMemory();
	}
}

void args_1(benchmark::internal::Benchmark* b) {
	isize ns[] = {64, 128, 255, 256, 1024};
	for (auto n : ns) {
		b->Arg(n);
	}
}
void args_2(benchmark::internal::Benchmark* b) {
	isize ns[] = {64, 128, 255, 256, 1024};
	isize rs[] = {1, 2, 3, 4, 8, 16, 32};
	for (auto n : ns) {
		for (auto r : rs) {
			b->Args({n, r});
		}
	}
}

// LDLT_BENCHMARK(bench_ours_r4)->Apply(args_2);
// LDLT_BENCHMARK(bench_ours__r)->Apply(args_2);
// LDLT_BENCHMARK(bench_eigen_r)->Apply(args_2);
LDLT_BENCHMARK(bench_ours___)->Apply(args_1);
LDLT_BENCHMARK(bench_eigen__)->Apply(args_1);

LDLT_BENCHMARK_MAIN();
