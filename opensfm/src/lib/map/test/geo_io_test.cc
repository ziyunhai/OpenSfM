#include <gtest/gtest.h>
#include <map/geo_io.h>

#include <cmath>
#include <string>
#include <vector>

namespace {

TEST(GeoIo, ParseBasicCsv) {
  const std::string content =
      "# image name,Easting [meter],Northing [meter],Gravity-related height "
      "[meter],yaw [degrees],pitch [degrees],roll [degrees],accuracy "
      "horizontal [meter],accuracy vertical [meter]\n"
      "img1.jpg,2681192.57,1250342.89,605.29,105.98,13.89,-6.94,0.03,0.05\n";

  std::vector<std::string> images = {"img1.jpg"};

  // CRS corresponds to CH1903 / LV95 -> EPSG:2056
  auto data = map::ParseGeolocationFile(content, images, "EPSG:2056");

  ASSERT_EQ(data.size(), 1u);
  EXPECT_EQ(data[0].filename, "img1.jpg");
  EXPECT_TRUE(data[0].has_lla);
  // Approx coordinates for 2681192.57, 1250342.89 in Zurich
  EXPECT_NEAR(data[0].lat, 47.404, 0.01);
  EXPECT_NEAR(data[0].lon, 8.510, 0.01);
  EXPECT_NEAR(data[0].alt, 605.29, 0.5);

  EXPECT_TRUE(data[0].has_std);
  EXPECT_DOUBLE_EQ(data[0].lat_std, 0.03);
  EXPECT_DOUBLE_EQ(data[0].lon_std, 0.03);
  EXPECT_DOUBLE_EQ(data[0].alt_std, 0.05);

  EXPECT_TRUE(data[0].has_ypr);
  EXPECT_DOUBLE_EQ(data[0].yaw, 105.98);
  EXPECT_DOUBLE_EQ(data[0].pitch, 13.89);
  EXPECT_DOUBLE_EQ(data[0].roll, -6.94);
}

TEST(GeoIo, ParseWgs84Identity) {
  const std::string content = "img2.jpg 8.51 47.404 600.0\n";

  std::vector<std::string> images = {"img2.jpg"};

  auto data = map::ParseGeolocationFile(content, images, "WGS84");

  ASSERT_EQ(data.size(), 1u);
  EXPECT_EQ(data[0].filename, "img2.jpg");
  EXPECT_TRUE(data[0].has_lla);
  EXPECT_DOUBLE_EQ(data[0].lat, 47.404);
  EXPECT_DOUBLE_EQ(data[0].lon, 8.51);
  EXPECT_DOUBLE_EQ(data[0].alt, 600.0);
}

}  // namespace