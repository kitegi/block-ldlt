include(../cmake/doctest.cmake)

include_directories(include)

add_library(backward_cpp_main OBJECT src/backward.cpp)
target_link_libraries(backward_cpp_main CONAN_PKG::backward-cpp)

add_library(doctest_main STATIC src/doctest_main.cpp)
add_library(libcnpy OBJECT src/cnpy.cpp)
target_link_libraries(libcnpy CONAN_PKG::eigen)
set(LDLT_TEST_LIBS
    block-ldlt
    doctest_main
    libcnpy
    ldlt_util
    backward_cpp_main
)

macro(ldlt_test testname testpath)
  add_executable(${testname} ${testpath})
  doctest_discover_tests(${testname})
  target_link_libraries(${testname} ${LDLT_TEST_LIBS})
  set(LDLT_TEST_LIST ${LDLT_TEST_LIST} ${testname})
endmacro()

ldlt_test(.test.factorize.roundtrip src/ldlt_roundtrip.cpp)
ldlt_test(.test.solve src/solve.cpp)
ldlt_test(.test.update.roundtrip src/update_roundtrip.cpp)
ldlt_test(.test.delete src/delete.cpp)
ldlt_test(.test.append src/append.cpp)

ldlt_test(.test.permute.solve src/permutation_solve.cpp)
ldlt_test(.test.permute.compute src/permutation_compute.cpp)
ldlt_test(.test.permute.apply src/permutation_apply.cpp)

ldlt_test(.test.wrapper.solve src/wrapper_solve.cpp)
ldlt_test(.test.wrapper.delete src/wrapper_delete.cpp)
ldlt_test(.test.wrapper.insert src/wrapper_insert.cpp)

add_custom_target(
  check
  COMMAND ${CMAKE_CTEST_COMMAND} --verbose
  DEPENDS ${LDLT_TEST_LIST}
)
