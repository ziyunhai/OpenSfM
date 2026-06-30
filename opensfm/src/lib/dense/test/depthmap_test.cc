#include <dense/fuser.h>
#include <gtest/gtest.h>

namespace {

using namespace dense;

// Type alias smoke test — ensures fuser.h compiles.
TEST(DenseTypes, ImageFAlias) {
  ImageF img(2, 3);
  EXPECT_EQ(img.rows(), 2);
  EXPECT_EQ(img.cols(), 3);
}

}  // namespace
