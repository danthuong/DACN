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



def test_depth_consistent_calib_scales_ground_depth_only():
    from lib.datasets.kitti.near_gpro_dataset import NearGProKITTIDataset

    p2 = np.array([[700.0, 0.0, 600.0, 45.0],
                   [0.0, 700.0, 180.0, -0.3],
                   [0.0, 0.0, 1.0, 0.003]])
    crop_scale = 1.04
    affine = np.array([[1.0 / crop_scale, 0.0, 12.0],
                       [0.0, 1.0 / crop_scale, -5.0],
                       [0.0, 0.0, 1.0]])
    input_p2 = affine @ p2
    virtual = NearGProKITTIDataset._depth_consistent_calib(input_p2, crop_scale)

    camera_height = 1.65
    ground_point = np.array([3.0, camera_height, 25.0, 1.0])
    pixel = input_p2 @ ground_point
    pixel = np.array([pixel[0] / pixel[2], pixel[1] / pixel[2], 1.0])

    inverse = np.linalg.inv(virtual[:, :3].astype(np.float64))
    center = -inverse @ virtual[:, 3]
    direction = inverse @ pixel
    distance = (camera_height - center[1]) / direction[1]
    hit = center + distance * direction
    # The projection's small translation column is scaled too, so allow cm.
    np.testing.assert_allclose(hit[2], 25.0 * crop_scale, atol=0.05)
    np.testing.assert_allclose(hit[0], 3.0, atol=0.05)
    np.testing.assert_allclose(
        NearGProKITTIDataset._depth_consistent_calib(input_p2, 1.0), input_p2,
        rtol=1e-6)
