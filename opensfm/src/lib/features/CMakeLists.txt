set(FEATURES_FILES
    akaze_bind.h
    hahog.h
    matching.h
    src/akaze_bind.cc
    src/hahog.cc
    src/matching.cc
)
add_library(features ${FEATURES_FILES})
target_link_libraries(features
  PRIVATE
    ${OpenCV_LIBS}
    akaze
    foundation
    vl
)
target_include_directories(features PRIVATE ${opensfm_SOURCE_DIR})

if (OPENSFM_BUILD_TESTS)
    set(FEATURES_TEST_FILES
        test/vlad_test.cc
    )
    add_executable(features_test ${FEATURES_TEST_FILES})
    target_include_directories(features_test PRIVATE ${opensfm_SOURCE_DIR})
    target_link_libraries(features_test
                        PRIVATE
                        features
                        foundation
                        Eigen3::Eigen
                        ${TEST_MAIN}
                        pybind11::embed
                        )
    add_test(features_test features_test)
endif()

pybind11_add_module(pyfeatures python/pybind.cc)
target_include_directories(pyfeatures PRIVATE ${GLOG_INCLUDE_DIR})
target_link_libraries(pyfeatures
  PRIVATE
    features
    foundation
    akaze
)
set_target_properties(pyfeatures PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/.."
)
install(TARGETS pyfeatures LIBRARY DESTINATION .)
