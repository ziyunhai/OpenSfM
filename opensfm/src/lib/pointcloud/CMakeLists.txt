set(POINTCLOUD_FILES 
    tile_io.cc
    octree_builder.cc
)

add_library(pointcloud ${POINTCLOUD_FILES})
target_include_directories(pointcloud 
  PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    ${opensfm_SOURCE_DIR}
)



pybind11_add_module(pypointcloud python/pybind.cc)
target_link_libraries(pypointcloud 
    PRIVATE 
        pointcloud
)

set_target_properties(pypointcloud PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/.."
)
install(TARGETS pypointcloud LIBRARY DESTINATION .)

if (OPENSFM_BUILD_TESTS)
    set(POINTCLOUD_TEST_FILES
        test/half_float_test.cc
        test/morton_test.cc
        test/tile_io_test.cc
    )
    add_executable(pointcloud_test ${POINTCLOUD_TEST_FILES})
    target_include_directories(pointcloud_test PRIVATE ${opensfm_SOURCE_DIR})
    target_link_libraries(pointcloud_test
                        PUBLIC
                        pointcloud
                        ${TEST_MAIN})
    add_test(pointcloud_test pointcloud_test)
endif()