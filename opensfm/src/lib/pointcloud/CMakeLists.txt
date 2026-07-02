set(POINTCLOUD_FILES
    tile_io.cc
    octree_builder.cc
    point_cloud_io.cc
    ply_io.cc
    las_io.cc
    laz_io.cc
    ooc_octree.cc
)

add_library(pointcloud ${POINTCLOUD_FILES})
target_compile_features(pointcloud PUBLIC cxx_std_17)
# LAS is hand-rolled (no external dependency); always available.
target_compile_definitions(pointcloud PUBLIC POINTCLOUD_HAVE_LAS=1)
target_include_directories(pointcloud
  PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    ${opensfm_SOURCE_DIR}
    ${opensfm_SOURCE_DIR}/third_party/csv2  # for bundled <csv2/csv2.hpp> (mio::mmap_source)
)

# ── LAZ: LASzip codec (hard-required; provided by conda-forge::laszip) ──
# Prefer the CMake package config (exports target laszip::api); fall back to a
# plain header+library search so different laszip packagings both work.
find_package(laszip QUIET)
if(TARGET laszip::api)
  target_link_libraries(pointcloud PUBLIC laszip::api)
elseif(TARGET laszip)
  target_link_libraries(pointcloud PUBLIC laszip)
else()
  find_path(LASZIP_INCLUDE_DIR laszip_api.h
    PATH_SUFFIXES laszip include/laszip include)
  find_library(LASZIP_LIBRARY NAMES laszip laszip_api laszip3 laszip_api3)
  if(NOT LASZIP_INCLUDE_DIR OR NOT LASZIP_LIBRARY)
    message(FATAL_ERROR
      "LASzip not found. Install conda-forge::laszip (it is a required "
      "dependency for LAZ point-cloud support).")
  endif()
  target_include_directories(pointcloud PUBLIC ${LASZIP_INCLUDE_DIR})
  target_link_libraries(pointcloud PUBLIC ${LASZIP_LIBRARY})
endif()
target_compile_definitions(pointcloud PUBLIC POINTCLOUD_HAVE_LAZ=1)



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