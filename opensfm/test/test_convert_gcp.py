# pyre-strict
import os
from typing import Any

from opensfm import io, dataset
from opensfm.actions import convert_gcp


def test_convert_gcp_txt_to_json(tmpdir: Any) -> None:
    data_dir = str(tmpdir.mkdir("dataset"))
    os.mkdir(os.path.join(data_dir, "images"))
    with open(os.path.join(data_dir, "images", "image1.jpg"), "w") as f:
        f.write("")
    data = dataset.DataSet(data_dir)

    image_name = "image1.jpg"
    exif_data = {"width": 1000, "height": 800, "camera": "cam1"}
    data.save_exif(image_name, exif_data)

    gcp_list_content = """WGS84
13.400502446\t52.519251158\t16.7021233002\t234.6\t462.8\timage1.jpg\tgcp1
"""
    gcp_list_path = data._gcp_list_file()
    with data.io_handler.open_wt(gcp_list_path) as fout:
        fout.write(gcp_list_content)

    convert_gcp.run_dataset(data, from_format="txt")

    json_path = data._ground_control_points_file()
    assert data.io_handler.isfile(json_path)

    with data.io_handler.open_rt(json_path) as fin:
        gcps, crs = io.read_ground_control_points(fin)

    assert crs == "WGS84"
    assert len(gcps) == 1
    assert gcps[0].id == "gcp1"
    assert len(gcps[0].observations) == 1
    assert gcps[0].observations[0].shot_id == "image1.jpg"

    # Test the backup functionality on the target json
    gcp_list_content_2 = """WGS84
13.400502446\t52.519251158\t16.7021233002\t234.6\t462.8\timage1.jpg\tgcp2
"""
    with data.io_handler.open_wt(gcp_list_path) as fout:
        fout.write(gcp_list_content_2)

    convert_gcp.run_dataset(data, from_format="txt")

    json_bak_path = json_path + ".bak"
    assert data.io_handler.isfile(json_bak_path)
    with data.io_handler.open_rt(json_bak_path) as fin:
        gcps_bak, _ = io.read_ground_control_points(fin)
    assert gcps_bak[0].id == "gcp1"

    with data.io_handler.open_rt(json_path) as fin:
        gcps_new, _ = io.read_ground_control_points(fin)
    assert gcps_new[0].id == "gcp2"


def test_convert_gcp_json_to_txt(tmpdir: Any) -> None:
    data_dir = str(tmpdir.mkdir("dataset"))
    os.mkdir(os.path.join(data_dir, "images"))
    with open(os.path.join(data_dir, "images", "image1.jpg"), "w") as f:
        f.write("")
    data = dataset.DataSet(data_dir)

    image_name = "image1.jpg"
    exif_data = {"width": 1000, "height": 800, "camera": "cam1"}
    data.save_exif(image_name, exif_data)

    json_content = """{
  "crs": "WGS84",
  "points": [
    {
      "id": "gcp1",
      "position": {
        "latitude": 52.519251158,
        "longitude": 13.400502446,
        "altitude": 16.7021233002
      },
      "observations": [
        {
          "shot_id": "image1.jpg",
          "projection": [0.2346, 0.4628]
        }
      ]
    }
  ]
}"""
    json_path = data._ground_control_points_file()
    with data.io_handler.open_wt(json_path) as fout:
        fout.write(json_content)

    convert_gcp.run_dataset(data, from_format="json")

    gcp_path = data._gcp_list_file()
    assert data.io_handler.isfile(gcp_path)

    with data.io_handler.open_rt(gcp_path) as fin:
        gcps = io.read_gcp_list(fin, {image_name: exif_data})

    assert len(gcps) == 1
    assert gcps[0].id == "gcp1"

    # Test the backup functionality on the target txt
    json_content_2 = json_content.replace("gcp1", "gcp2")
    with data.io_handler.open_wt(json_path) as fout:
        fout.write(json_content_2)

    convert_gcp.run_dataset(data, from_format="json")

    gcp_bak_path = gcp_path + ".bak"
    assert data.io_handler.isfile(gcp_bak_path)
    with data.io_handler.open_rt(gcp_bak_path) as fin:
        gcps_bak = io.read_gcp_list(fin, {image_name: exif_data})
    assert gcps_bak[0].id == "gcp1"

    with data.io_handler.open_rt(gcp_path) as fin:
        gcps_new = io.read_gcp_list(fin, {image_name: exif_data})
    assert gcps_new[0].id == "gcp2"
