# Common CMake configuration for OpenSfM

# Visibility stuff
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_INLINES ON)

# fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable all warnings
if (NOT WIN32)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-sign-compare")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-compare")
endif()

# Important defines
add_definitions(-DVL_DISABLE_AVX)
add_definitions(-DGLOG_USE_GLOG_EXPORT)
add_definitions(-DINPLACE_VLFEAT)

if(NOT USE_SSE2)
  add_definitions(-DVL_DISABLE_SSE2)
endif()

if (WIN32)
    # Missing math constant
    add_definitions(-DM_PI=3.14159265358979323846)
endif()

####### Find Common Dependencies #######
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Eigen3 REQUIRED)
find_package(Gflags REQUIRED)

# glog finding logic
find_package(glog QUIET CONFIG)
if(glog_FOUND)
  message(STATUS "Using glog native CMake config (modern glog)")
else()
  message(STATUS
    "glog CMake config not found, using FindGlog.cmake "
    "(Ubuntu 20.04 compatibility)"
  )
  find_package(Glog REQUIRED)
endif()
