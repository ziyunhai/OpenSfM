cmake_minimum_required(VERSION 3.15)
project(opensfm_root)

option(OPENSFM_BUILD_TESTS "Build OpenSfM unit tests." on)

if(OPENSFM_BUILD_TESTS)
	enable_testing()
endif()

add_subdirectory(lib)

