include_directories(include)

set(LDLT_BENCH_LIBS block-ldlt ldlt_util CONAN_PKG::benchmark)

add_executable(.bench.factorize src/factorize.cpp)
target_link_libraries(.bench.factorize ${LDLT_BENCH_LIBS})
add_executable(.bench.rank_update src/rank_update.cpp)
target_link_libraries(.bench.rank_update ${LDLT_BENCH_LIBS})
target_include_directories(.bench.rank_update PUBLIC ${BACKWARD_INCLUDE_DIRS})
