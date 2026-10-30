import numpy as np

from generate_near_gpro_pseudo_labels import add_near_contact


def test_near_contact_is_midpoint_of_two_smallest_camera_z_candidates():
    record = {
        "image_size": [1280, 384],
        "calibration": {
            "P2": [[100.0, 0.0, 640.0, 0.0],
                   [0.0, 100.0, 192.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0]],
        },
        "objects": [{
            "object_id": 3,
            "touch_points": [
                {"name": "LF", "local_3d": [1.0, 0.0, -1.0],
                 "camera_3d": [1.0, 1.5, 8.0]},
                {"name": "RF", "local_3d": [-1.0, 0.0, -1.0],
                 "camera_3d": [-1.0, 1.5, 10.0]},
                {"name": "RR", "local_3d": [-1.0, 0.0, 1.0],
                 "camera_3d": [-1.0, 1.5, 14.0]},
                {"name": "LR", "local_3d": [1.0, 0.0, 1.0],
                 "camera_3d": [1.0, 1.5, 12.0]},
            ],
        }],
    }
    assert add_near_contact(record) == 1
    near = record["objects"][0]["near_contact"]
    assert near["selected_names"] == ["LF", "RF"]
    np.testing.assert_allclose(near["camera_3d"], [0.0, 1.5, 9.0])
    np.testing.assert_allclose(near["local_offset_3d"], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(near["pixel_2d"], [640.0, 192.0 + 100.0 / 6.0])

