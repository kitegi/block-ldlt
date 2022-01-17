cmake_minimum_required(VERSION 3.16)

get_directory_property(has_parent PARENT_DIRECTORY)

if(has_parent)
  set(TOP_LEVEL FALSE)
else()
  set(TOP_LEVEL TRUE)
endif()

project(block-ldlt CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

add_library(
  block-ldlt
  ${CMAKE_CURRENT_SOURCE_DIR}/src/eigen_f32.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/eigen_f64.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/factorize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/solve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/update.cpp
)
target_include_directories(
  block-ldlt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

option(BLOCK_LDLT_TESTS "enable tests" OFF)

if(BLOCK_LDLT_TESTS)
  add_library(project_warnings INTERFACE)
  add_library(project_options INTERFACE)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler_warnings.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/static_analyzers.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers.cmake)
  target_link_libraries(block-ldlt INTERFACE project_options project_warnings)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extra_local_settings.cmake)
  enable_sanitizers(project_options)
  set_project_warnings(project_warnings)

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/conan.cmake)
  # Conan dependencies
  set(CONAN_REQUIRES
      fmt/8.0.1
      backward-cpp/1.5
      benchmark/1.6.0
      eigen/3.3.7
  )
  set(CONAN_OPTIONS)
  run_conan()
  target_link_libraries(block-ldlt PUBLIC veg CONAN_PKG::fmt CONAN_PKG::eigen)
else()
  find_package(Eigen3 REQUIRED)
  target_link_libraries(block-ldlt PUBLIC veg Eigen3::Eigen)
endif()

add_subdirectory(external/veg)
target_include_directories(
  block-ldlt SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/simde/
)

if(BLOCK_LDLT_TESTS)
  add_library(ldlt_util STATIC test/src/util_f64.cpp test/src/util_f32.cpp)
  target_include_directories(ldlt_util PUBLIC test/include)
  target_link_libraries(ldlt_util block-ldlt)

  enable_testing()
  add_subdirectory(test)

  add_subdirectory(benchmark)
endif()

include(GNUInstallDirs)
install(
  TARGETS block-ldlt
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include/)
