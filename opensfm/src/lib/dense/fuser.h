#pragma once

#include <foundation/types.h>

#include <Eigen/Core>
#include <cstdint>

namespace dense {

// ---- Image-like type aliases for the dense module ----
// Row-major storage matches cv::Mat / numpy (C-order) memory layout.
using ImageF =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageI =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageU8 =
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Per-pixel 3-component data stored column-wise: each column is one pixel's
// Vec3{f,u8}.  Shape: (3, H*W).  ColMajor storage gives interleaved layout
// [x0,y0,z0, x1,y1,z1, ...] matching cv::Mat CV_32FC3 / CV_8UC3.
using PixelData3f = Eigen::Matrix<float, 3, Eigen::Dynamic>;
using PixelData3u8 = Eigen::Matrix<uint8_t, 3, Eigen::Dynamic>;

}  // namespace dense
