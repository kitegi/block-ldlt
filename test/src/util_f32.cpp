#include <util.hpp>

namespace ldlt_test {
namespace eigen {
LDLT_EXPLICIT_TPL_DEF(2, llt_compute<Mat<f32, colmajor>>);
LDLT_EXPLICIT_TPL_DEF(2, ldlt_compute<Mat<f32, colmajor>>);
LDLT_EXPLICIT_TPL_DEF(2, llt_compute<Mat<f32, rowmajor>>);
LDLT_EXPLICIT_TPL_DEF(2, ldlt_compute<Mat<f32, rowmajor>>);
} // namespace eigen
namespace rand {
LDLT_EXPLICIT_TPL_DEF(2, matrix_rand<f32>);
LDLT_EXPLICIT_TPL_DEF(1, vector_rand<f32>);
LDLT_EXPLICIT_TPL_DEF(2, positive_definite_rand<f32>);
LDLT_EXPLICIT_TPL_DEF(1, orthonormal_rand<f32>);
LDLT_EXPLICIT_TPL_DEF(3, sparse_matrix_rand<f32>);
LDLT_EXPLICIT_TPL_DEF(3, sparse_positive_definite_rand<f32>);
} // namespace rand
} // namespace ldlt_test

LDLT_EXPLICIT_TPL_DEF(2, matmul_impl<long double>);
LDLT_EXPLICIT_TPL_DEF(1, mat_cast<ldlt::f32, long double>);
