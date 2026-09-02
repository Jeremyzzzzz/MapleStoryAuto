import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.yolo_monster_viewer import (
    AsyncNameOcrLocator,
    DEFAULT_WINDOW_TOKEN,
    DetectionTracker,
    EntityCoordinateTracker,
    REQUIRED_CLASSES,
    ReadOnlyPlayerDetector,
    attach_player_relative_coordinates,
    draw_detections,
    deduplicate_detections,
    normalize_label,
    normalize_model_class,
    reject_edge_clipped_stump,
    locate_minimap_players,
    locate_minimap_player,
    MinimapRedMarkerTracker,
    resolve_gameplay_height,
    resolve_minimap_canvas,
    resolve_scaled_region,
    serialize_minimap_player,
    validate_model_classes,
)


class YoloMonsterViewerTests(unittest.TestCase):
    @staticmethod
    def detection(monster_class, box, confidence=0.9):
        return {
            "class": monster_class,
            "label": monster_class.upper(),
            "label_zh": monster_class,
            "confidence": confidence,
            "box": box,
            "color": (1, 2, 3),
        }

    def test_default_window_title_is_readable(self):
        self.assertEqual(DEFAULT_WINDOW_TOKEN, "冒险岛怀旧服")

    def test_gameplay_height_scales_up_with_wider_capture(self):
        self.assertEqual(
            resolve_gameplay_height((1127, 1924, 3), 687, 1370),
            965,
        )
        self.assertEqual(
            resolve_gameplay_height((759, 1296, 3), 687, 1370),
            687,
        )

    def test_minimap_region_scales_and_clamps_to_capture(self):
        self.assertEqual(
            resolve_scaled_region((815, 1370, 3), [23, 117, 125, 137], 1370),
            [23, 117, 125, 137],
        )
        self.assertEqual(
            resolve_scaled_region((1127, 1924, 3), [23, 117, 125, 137], 1370),
            [32, 164, 176, 192],
        )

    def test_minimap_canvas_auto_width_follows_expanded_panel_border(self):
        frame = np.zeros((260, 300, 3), dtype=np.uint8)
        frame[100:180, 140:143] = (255, 255, 255)
        frame[180:183, 17:143] = (220, 220, 220)
        frame[20:250, 243:246] = (255, 255, 255)
        canvas = resolve_minimap_canvas(
            frame,
            [20, 100, 120, 130],
            {
                "canvas_auto_width": True,
                "region": [0, 0, 285, 245],
                "canvas_reference_width": 300,
            },
        )

        self.assertEqual(canvas, [20, 100, 120, 80])

    def test_minimap_player_detection_uses_compact_yellow_component(self):
        frame = np.zeros((260, 300, 3), dtype=np.uint8)
        frame[20:30, 20:30] = (0, 255, 255)
        frame[170, 91:95] = (0, 255, 255)
        frame[169, 92:94] = (0, 255, 255)
        result = locate_minimap_player(
            frame,
            {
                "canvas_region": [20, 100, 120, 130],
                "canvas_reference_width": 300,
                "player_color": [0, 255, 255],
                "marker_color_tolerance": [40, 40, 40],
            },
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["marker_box_map"], [71, 69, 4, 2])
        self.assertEqual(result["map_px"], [72.5, 69.5])
        self.assertEqual(result["frame_px"], [92.5, 169.5])
        self.assertAlmostEqual(result["map_norm"][0], 72.5 / 120.0)

    def test_minimap_player_detection_ignores_yellow_outside_canvas(self):
        frame = np.zeros((260, 300, 3), dtype=np.uint8)
        frame[20:30, 20:30] = (0, 255, 255)
        result = locate_minimap_player(
            frame,
            {
                "canvas_region": [100, 100, 80, 80],
                "canvas_reference_width": 300,
                "player_color": [0, 255, 255],
                "marker_color_tolerance": [40, 40, 40],
            },
        )

        self.assertIsNone(result)

    def test_minimap_detects_all_red_other_players_with_coordinates(self):
        frame = np.zeros((260, 300, 3), dtype=np.uint8)
        # The yellow self marker and two red player dots are inside the
        # calibrated canvas. A larger red UI block outside the canvas must be
        # ignored because detection never searches the surrounding title/UI.
        frame[169:172, 92:95] = (0, 255, 255)
        cv2.circle(frame, (45, 120), 2, (0, 0, 255), -1)
        cv2.circle(frame, (100, 175), 2, (0, 0, 255), -1)
        frame[20:30, 20:30] = (0, 0, 255)
        result = locate_minimap_players(
            frame,
            {
                "canvas_region": [20, 100, 120, 130],
                "canvas_reference_width": 300,
                "player_color": [0, 255, 255],
                "marker_color_tolerance": [40, 40, 40],
                "other_player_color": [0, 0, 255],
                "other_player_marker_color_tolerance": [45, 45, 45],
            },
        )

        self.assertIsNotNone(result["player"])
        self.assertEqual(len(result["other_players"]), 2)
        coords = [tuple(marker["map_px"]) for marker in result["other_players"]]
        self.assertEqual(coords, [(25.0, 20.0), (80.0, 75.0)])
        self.assertEqual(
            [tuple(marker["frame_px"]) for marker in result["other_players"]],
            [(45.0, 120.0), (100.0, 175.0)],
        )

    def test_minimap_red_marker_filter_rejects_large_or_thin_components(self):
        frame = np.zeros((220, 220, 3), dtype=np.uint8)
        frame[100:103, 50:53] = (0, 0, 255)  # valid 3x3 dot
        frame[120:135, 80:95] = (0, 0, 255)  # too large
        frame[150:170, 110:112] = (0, 0, 255)  # too thin
        result = locate_minimap_players(
            frame,
            {
                "canvas_region": [20, 80, 160, 120],
                "canvas_reference_width": 220,
                "other_player_color": [0, 0, 255],
                "other_player_marker_color_tolerance": [45, 45, 45],
            },
        )
        self.assertEqual(len(result["other_players"]), 1)
        self.assertEqual(result["other_players"][0]["map_px"], [31.0, 21.0])

    def test_minimap_accepts_two_by_three_red_player_dot(self):
        frame = np.zeros((180, 180, 3), dtype=np.uint8)
        frame[100:103, 52:54] = (0, 0, 238)
        result = locate_minimap_players(
            frame,
            {
                "canvas_region": [20, 80, 120, 80],
                "canvas_reference_width": 180,
                "other_player_hsv_ranges": [[0, 170, 175, 12, 255, 255]],
                "other_player_red_min": 175,
                "other_player_red_green_gap": 120,
                "other_player_red_blue_gap": 120,
                "other_player_marker_min_pixels": 4,
                "other_player_marker_max_pixels": 30,
                "other_player_marker_min_dimension": 2,
                "other_player_marker_max_dimension": 8,
                "other_player_marker_min_fill_ratio": 0.35,
            },
        )
        self.assertEqual(len(result["other_players"]), 1)
        self.assertEqual(result["other_players"][0]["marker_box_map"], [32, 20, 2, 3])

    def test_minimap_red_tracker_requires_two_consecutive_frames(self):
        marker = {
            "map_px": [30.0, 20.0],
            "map_norm": [0.3, 0.2],
        }
        tracker = MinimapRedMarkerTracker(confirm_frames=2, max_missed=1)
        self.assertEqual(tracker.update([marker]), [])
        self.assertEqual(tracker.update([marker]), [marker])
        # A one-frame dropout is tolerated, but a new isolated dot is not
        # exposed immediately.
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([marker]), [marker])

    def test_minimap_player_serialization_contains_plain_values(self):
        serialized = serialize_minimap_player(
            {
                "map_px": np.array([72.5, 69.5], dtype=np.float32),
                "map_norm": np.array([0.604, 0.535], dtype=np.float32),
                "frame_px": np.array([92.5, 169.5], dtype=np.float32),
                "marker_box_map": np.array([71, 69, 4, 2], dtype=np.int64),
                "canvas_frame_box": np.array([20, 100, 120, 130], dtype=np.int64),
                "canvas_size": np.array([120, 130], dtype=np.int64),
                "pixel_count": np.int64(6),
                "fill_ratio": np.float32(0.75),
            }
        )
        self.assertEqual(serialized["map_px"], [72.5, 69.5])
        self.assertTrue(all(type(value) is int for value in serialized["canvas_size"]))

    def test_accepts_all_requested_chinese_labels(self):
        labels = [
            normalize_label(label)
            for label in (
                "树妖",
                "红蜗牛",
                "绿水灵",
                "绿蘑菇",
                "花蘑菇",
                "僵尸蘑菇",
                "刺蘑菇",
            )
        ]

        self.assertEqual(set(labels), REQUIRED_CLASSES)

    def test_green_slime_model_alias_maps_to_slime(self):
        self.assertEqual(normalize_model_class("green_slime"), "slime")

    def test_wooden_stump_chinese_alias_maps_to_stump(self):
        self.assertEqual(normalize_label("木妖"), "stump")

    def test_rejects_model_missing_new_mushroom_classes(self):
        names = {
            0: "stump",
            1: "red_snail",
            2: "slime",
            3: "green_mushroom",
            4: "flower_mushroom",
        }

        with self.assertRaisesRegex(
            ValueError, "thorn_mushroom.*zombie_mushroom"
        ):
            validate_model_classes(names)

    def test_accepts_exact_seven_class_model(self):
        names = {
            0: "slime",
            1: "red_snail",
            2: "green_mushroom",
            3: "stump",
            4: "flower_mushroom",
            5: "zombie_mushroom",
            6: "thorn_mushroom",
        }

        validate_model_classes(names)

    def test_accepts_dedicated_two_class_model_for_selected_labels(self):
        names = {0: "zombie_mushroom", 1: "thorn_mushroom"}

        validate_model_classes(
            names, {"zombie_mushroom", "thorn_mushroom"}
        )

    def test_deduplicates_overlapping_same_class_boxes(self):
        detections = [
            {
                "class": "flower_mushroom",
                "confidence": 0.95,
                "box": [100, 100, 60, 60],
            },
            {
                "class": "flower_mushroom",
                "confidence": 0.70,
                "box": [103, 102, 62, 61],
            },
        ]

        kept = deduplicate_detections(detections)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["confidence"], 0.95)

    def test_rejects_short_box_clipped_at_gameplay_bottom(self):
        self.assertTrue(reject_edge_clipped_stump([671, 633, 63, 54], 687))

    def test_keeps_normal_height_stump_at_gameplay_bottom(self):
        self.assertFalse(reject_edge_clipped_stump([724, 604, 66, 80], 687))

    def test_keeps_short_stump_away_from_gameplay_bottom(self):
        self.assertFalse(reject_edge_clipped_stump([700, 580, 60, 54], 687))

    def test_keeps_overlapping_boxes_from_different_classes(self):
        detections = [
            {"class": "stump", "confidence": 0.9, "box": [10, 10, 40, 50]},
            {
                "class": "green_mushroom",
                "confidence": 0.8,
                "box": [10, 10, 40, 50],
            },
        ]

        self.assertEqual(len(deduplicate_detections(detections)), 2)

    def test_tracker_smooths_and_preserves_track_id(self):
        tracker = DetectionTracker(smoothing=0.5)
        first = tracker.update([self.detection("stump", [100, 100, 40, 50])])
        second = tracker.update([self.detection("stump", [110, 100, 40, 50])])

        self.assertEqual(second[0]["track_id"], first[0]["track_id"])
        self.assertGreater(second[0]["box"][0], 100)
        self.assertLess(second[0]["box"][0], 110)
        self.assertEqual(second[0]["missed_frames"], 0)

    def test_tracker_bridges_short_detection_dropout(self):
        tracker = DetectionTracker(max_missed=2)
        detected = tracker.update(
            [self.detection("green_mushroom", [20, 30, 32, 40])]
        )
        first_miss = tracker.update([])
        second_miss = tracker.update([])
        expired = tracker.update([])

        self.assertEqual(first_miss[0]["track_id"], detected[0]["track_id"])
        self.assertEqual(first_miss[0]["missed_frames"], 1)
        self.assertEqual(second_miss[0]["missed_frames"], 2)
        self.assertEqual(expired, [])

    def test_tracker_does_not_match_different_classes(self):
        tracker = DetectionTracker()
        first = tracker.update([self.detection("stump", [50, 50, 40, 50])])
        second = tracker.update(
            [self.detection("green_mushroom", [50, 50, 40, 50])]
        )

        self.assertEqual(len(second), 2)
        self.assertNotEqual(second[0]["track_id"], second[1]["track_id"])
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])

    def test_tracker_reassociates_after_one_missed_frame(self):
        tracker = DetectionTracker(max_missed=2, max_center_distance=1.25)
        first = tracker.update([self.detection("slime", [100, 80, 40, 35])])
        tracker.update([])
        returned = tracker.update(
            [self.detection("slime", [120, 80, 40, 35])]
        )

        self.assertEqual(len(returned), 1)
        self.assertEqual(returned[0]["track_id"], first[0]["track_id"])
        self.assertEqual(returned[0]["missed_frames"], 0)

    def test_tracker_accepts_moderate_pose_size_change(self):
        tracker = DetectionTracker(smoothing=1.0)
        first = tracker.update(
            [self.detection("thorn_mushroom", [100, 100, 40, 50])]
        )
        changed = tracker.update(
            [self.detection("thorn_mushroom", [91, 85, 58, 80])]
        )

        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["track_id"], first[0]["track_id"])
        self.assertEqual(changed[0]["box"], [91, 85, 58, 80])

    def test_tracker_rejects_implausible_same_class_size_jump(self):
        tracker = DetectionTracker(smoothing=1.0)
        first = tracker.update(
            [self.detection("zombie_mushroom", [100, 100, 40, 50])]
        )
        changed = tracker.update(
            [self.detection("zombie_mushroom", [82, 72, 76, 105])]
        )

        self.assertEqual(len(changed), 2)
        self.assertEqual(changed[0]["track_id"], first[0]["track_id"])
        self.assertNotEqual(changed[1]["track_id"], first[0]["track_id"])
        self.assertEqual(changed[0]["missed_frames"], 1)
        self.assertEqual(changed[1]["missed_frames"], 0)

    def test_tracker_requires_two_hits_for_low_confidence_track(self):
        tracker = DetectionTracker(
            min_confirmed_hits=2,
            high_confidence_confirm=0.75,
        )

        first = tracker.update(
            [self.detection("thorn_mushroom", [100, 80, 40, 50], 0.30)]
        )
        second = tracker.update(
            [self.detection("thorn_mushroom", [103, 80, 40, 50], 0.35)]
        )

        self.assertEqual(first, [])
        self.assertEqual(len(second), 1)
        self.assertTrue(second[0]["confirmed"])
        self.assertEqual(second[0]["confirmation_hits"], 2)

    def test_tracker_confirms_high_confidence_on_first_frame(self):
        tracker = DetectionTracker(
            min_confirmed_hits=2,
            high_confidence_confirm=0.75,
        )

        detected = tracker.update(
            [self.detection("zombie_mushroom", [20, 30, 40, 50], 0.90)]
        )

        self.assertEqual(len(detected), 1)
        self.assertTrue(detected[0]["confirmed"])
        self.assertEqual(detected[0]["confirmation_hits"], 1)

    def test_unconfirmed_track_never_emits_predicted_box(self):
        tracker = DetectionTracker(
            max_missed=2,
            min_confirmed_hits=2,
            high_confidence_confirm=0.75,
        )

        tracker.update(
            [self.detection("thorn_mushroom", [100, 80, 40, 50], 0.20)]
        )

        self.assertEqual(tracker.update([]), [])

    def test_low_confidence_hits_must_be_consecutive(self):
        tracker = DetectionTracker(
            max_missed=2,
            min_confirmed_hits=2,
            high_confidence_confirm=0.75,
        )
        detection = self.detection(
            "thorn_mushroom", [100, 80, 40, 50], 0.20
        )

        self.assertEqual(tracker.update([detection]), [])
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([detection]), [])
        confirmed = tracker.update([detection])

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["confirmation_hits"], 2)

    def test_coordinate_tracker_reports_upward_motion(self):
        coordinates = EntityCoordinateTracker(velocity_smoothing=1.0)
        first = {
            **self.detection("slime", [100, 120, 40, 40]),
            "track_id": 7,
            "missed_frames": 0,
        }
        second = {
            **self.detection("slime", [100, 90, 40, 40]),
            "track_id": 7,
            "missed_frames": 0,
        }

        coordinates.update([first], 10.0, 200, 200)
        updated = coordinates.update([second], 10.2, 200, 200)[0]

        self.assertEqual(updated["entity_id"], "M7")
        self.assertEqual(updated["center_px"], [120.0, 110.0])
        self.assertEqual(updated["center_norm"], [0.6, 0.55])
        self.assertEqual(updated["motion_state"], "UP")
        self.assertLess(updated["velocity_px_s"][1], 0)

    def test_monster_coordinates_include_player_relative_delta(self):
        monster = {
            "entity_id": "M2",
            "center_px": [160.0, 130.0],
        }
        player = {"entity_id": "P1", "center_px": [100.0, 100.0]}

        updated = attach_player_relative_coordinates([monster], player)[0]

        self.assertEqual(updated["relative_to_player"]["delta_px"], [60.0, 30.0])
        self.assertAlmostEqual(
            updated["relative_to_player"]["distance_px"], 67.1, places=1
        )

    def test_player_detector_uses_nametag_position(self):
        rng = np.random.default_rng(7)
        template = rng.integers(0, 256, (9, 13, 3), dtype=np.uint8)
        frame = np.zeros((100, 140, 3), dtype=np.uint8)
        frame[60:69, 50:63] = template
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player.png"
            ok, encoded = cv2.imencode(".png", template)
            self.assertTrue(ok)
            encoded.tofile(str(path))
            detector = ReadOnlyPlayerDetector(
                path,
                threshold=0.95,
                box_size=(20, 30),
                center_offset_y=10,
            )
            result = detector.detect(frame, gameplay_height=90)

        self.assertIsNotNone(result)
        self.assertEqual(result["nametag_box"], [50, 60, 13, 9])
        self.assertEqual(result["box"], [46, 35, 20, 30])

    def test_player_detector_uses_blue_title_anchor_with_red_body(self):
        template = np.full((23, 41, 3), 40, dtype=np.uint8)
        frame = np.full((200, 260, 3), 10, dtype=np.uint8)
        # Long blue title strip below the name, plus the player's red wing/body
        # at the inferred center. This is the no-OCR anchor seen in the live
        # screenshot; unrelated blue UI is intentionally absent.
        frame[100:113, 100:211] = (255, 0, 0)
        frame[10:90, 130:180] = (0, 0, 220)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player.png"
            ok, encoded = cv2.imencode(".png", template)
            self.assertTrue(ok)
            encoded.tofile(str(path))
            detector = ReadOnlyPlayerDetector(
                path,
                threshold=0.99,
                identity_threshold=0.90,
                local_identity_threshold=0.90,
                max_valid_y=160,
                color_anchor_enabled=True,
            )
            result = detector.detect(frame, gameplay_height=180)

        self.assertIsNotNone(result)
        self.assertEqual(result["identity_mode"], "color_anchor")
        self.assertEqual(result["anchor_box"][1], 100)
        self.assertGreaterEqual(result["anchor_box"][2], 111)
        self.assertEqual(result["anchor_box"][3], 13)
        self.assertEqual(result["nametag_box"], [135, 76, 41, 23])
        self.assertGreater(result["red_fraction"], 0.20)

    def test_player_detector_rejects_isolated_blue_ui_anchor(self):
        template = np.full((23, 41, 3), 40, dtype=np.uint8)
        frame = np.full((200, 260, 3), 10, dtype=np.uint8)
        frame[100:113, 100:211] = (255, 0, 0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player.png"
            ok, encoded = cv2.imencode(".png", template)
            self.assertTrue(ok)
            encoded.tofile(str(path))
            detector = ReadOnlyPlayerDetector(
                path,
                threshold=0.99,
                identity_threshold=0.90,
                local_identity_threshold=0.90,
                max_valid_y=160,
                color_anchor_enabled=True,
            )
            result = detector.detect(frame, gameplay_height=180)

        self.assertIsNone(result)

    def test_player_detector_reranks_background_match_with_name_glyphs(self):
        rng = np.random.default_rng(4)
        template = np.full((23, 41, 3), 70, dtype=np.uint8)
        cv2.rectangle(template, (0, 0), (40, 22), (245, 245, 245), 1)
        cv2.putText(
            template,
            "ABC",
            (4, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (180, 180, 180),
            1,
            cv2.LINE_8,
        )
        correct_name = template.copy()
        dark_pixels = correct_name[:, :, 0] < 130
        noise = rng.integers(-20, 21, correct_name.shape[:2])
        noisy_gray = np.clip(
            correct_name[:, :, 0].astype(int) + noise, 0, 120
        ).astype(np.uint8)
        correct_name[dark_pixels] = np.repeat(
            noisy_gray[:, :, None], 3, axis=2
        )[dark_pixels]
        correct_name[[0, -1], :] = 80
        correct_name[:, [0, -1]] = 80

        wrong_name = template.copy()
        wrong_name[4:18, 3:38] = 70
        cv2.putText(
            wrong_name,
            "XYZ",
            (4, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (180, 180, 180),
            1,
            cv2.LINE_8,
        )
        wrong_template_score = cv2.matchTemplate(
            wrong_name, template, cv2.TM_CCOEFF_NORMED
        )[0, 0]
        correct_template_score = cv2.matchTemplate(
            correct_name, template, cv2.TM_CCOEFF_NORMED
        )[0, 0]
        self.assertGreater(wrong_template_score, correct_template_score)

        frame = np.full((100, 160, 3), 10, dtype=np.uint8)
        frame[15:38, 15:56] = wrong_name
        frame[55:78, 70:111] = correct_name
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player.png"
            ok, encoded = cv2.imencode(".png", template)
            self.assertTrue(ok)
            encoded.tofile(str(path))
            detector = ReadOnlyPlayerDetector(
                path,
                threshold=0.30,
                identity_threshold=0.45,
                identity_margin=0.0,
                center_weight=0.0,
            )
            result = detector.detect(frame, gameplay_height=90)

        self.assertIsNotNone(result)
        self.assertEqual(result["nametag_box"], [70, 55, 41, 23])
        self.assertGreater(result["glyph_score"], 0.95)
        self.assertEqual(result["identity_mode"], "global")

    def test_player_detector_does_not_switch_to_distant_name_while_locked(self):
        template = np.full((9, 13, 3), 40, dtype=np.uint8)
        template[2:7, 3:5] = 220
        template[2:7, 8:10] = 220
        first_frame = np.zeros((100, 160, 3), dtype=np.uint8)
        first_frame[55:64, 70:83] = template
        distant_frame = np.zeros((100, 160, 3), dtype=np.uint8)
        distant_frame[10:19, 10:23] = template

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player.png"
            ok, encoded = cv2.imencode(".png", template)
            self.assertTrue(ok)
            encoded.tofile(str(path))
            detector = ReadOnlyPlayerDetector(
                path,
                threshold=0.95,
                identity_threshold=0.95,
                identity_margin=0.0,
                lock_radius=30,
                reacquire_misses=3,
                center_weight=0.0,
            )
            locked = detector.detect(first_frame, gameplay_height=90)
            switched = detector.detect(distant_frame, gameplay_height=90)

        self.assertIsNotNone(locked)
        self.assertIsNone(switched)
        self.assertEqual(detector.last_location, (70, 55))
        self.assertEqual(detector.identity_misses, 1)

    def test_player_detector_waits_for_required_identity_seed(self):
        template = np.full((9, 13, 3), 40, dtype=np.uint8)
        template[2:7, 3:10] = 220
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[55:64, 70:83] = template

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player.png"
            ok, encoded = cv2.imencode(".png", template)
            self.assertTrue(ok)
            encoded.tofile(str(path))
            detector = ReadOnlyPlayerDetector(
                path,
                threshold=0.95,
                identity_threshold=0.95,
                local_identity_threshold=0.90,
                require_identity_seed=True,
            )
            before_seed = detector.detect(frame, gameplay_height=90)
            detector.seed_identity((70, 55))
            after_seed = detector.detect(frame, gameplay_height=90)

        self.assertIsNone(before_seed)
        self.assertIsNotNone(after_seed)
        self.assertEqual(after_seed["nametag_box"], [70, 55, 13, 9])
        self.assertEqual(after_seed["identity_mode"], "local")

    def test_player_detector_tracks_ocr_title_anchor_between_refreshes(self):
        rng = np.random.default_rng(12)
        name_template = rng.integers(0, 256, (9, 13, 3), dtype=np.uint8)
        title_template = rng.integers(0, 256, (7, 20, 3), dtype=np.uint8)
        first = np.zeros((100, 160, 3), dtype=np.uint8)
        first[50:57, 60:80] = title_template
        second = np.zeros((100, 160, 3), dtype=np.uint8)
        second[70:77, 90:110] = title_template

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player.png"
            ok, encoded = cv2.imencode(".png", name_template)
            self.assertTrue(ok)
            encoded.tofile(str(path))
            detector = ReadOnlyPlayerDetector(path, threshold=0.99)
            detector.seed_identity(
                {
                    "location": (65, 35),
                    "anchor_box": [60, 50, 20, 7],
                    "ocr_score": 0.99,
                },
                first,
            )
            result = detector.detect(second, gameplay_height=100)

        self.assertIsNotNone(result)
        self.assertEqual(result["nametag_box"], [95, 55, 13, 9])
        self.assertEqual(result["identity_mode"], "ocr_anchor")

    def test_ocr_identity_locator_selects_exact_player_name(self):
        locator = AsyncNameOcrLocator.__new__(AsyncNameOcrLocator)
        locator.player_name = "麻超圆"
        locator.template_height = 23
        locator.template_width = 41
        locator.confidence = 0.70
        entries = [
            [
                [[398, 530], [451, 530], [451, 543], [398, 543]],
                "甜虾在新",
                0.999,
            ],
            [
                [[684, 468], [725, 468], [725, 484], [684, 484]],
                "麻超圆",
                0.998,
            ],
        ]

        result = locator._locate_exact_name((entries, 0.1), frame_id=7)

        self.assertIsNotNone(result)
        self.assertEqual(result["location"], (684, 464))
        self.assertEqual(result["nametag_box"], [684, 468, 41, 16])
        self.assertEqual(result["frame_id"], 7)
        self.assertEqual(result["text"], "麻超圆")

    def test_ocr_title_anchor_rejects_unpaired_status_bar_name(self):
        locator = AsyncNameOcrLocator.__new__(AsyncNameOcrLocator)
        locator.player_name = "麻超圆"
        locator.template_height = 23
        locator.template_width = 41
        locator.confidence = 0.70
        locator.title_text = "中级冒险家勋章"
        locator.title_to_name_offset_y = 27
        entries = [
            [
                [[1777, 911], [1810, 911], [1810, 925], [1777, 925]],
                "麻超圆",
                0.995,
            ],
            [
                [[200, 730], [340, 730], [340, 753], [200, 753]],
                "中级冒险家勋章",
                0.980,
            ],
        ]

        result = locator._locate_exact_name((entries, 0.1), frame_id=8)

        self.assertIsNotNone(result)
        self.assertEqual(result["identity_source"], "ocr_title")
        self.assertEqual(result["anchor_box"], [200, 730, 140, 23])
        self.assertEqual(result["location"], (246, 703))

    def test_coordinate_render_adds_non_overlapping_side_panel(self):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        detection = {
            **self.detection("slime", [20, 30, 40, 40]),
            "entity_id": "M1",
            "center_px": [40.0, 50.0],
            "center_norm": [0.22222, 0.41667],
            "velocity_px_s": [10.0, -5.0],
            "speed_px_s": 11.2,
            "motion_state": "STILL",
            "tracking_state": "DETECTED",
            "track_id": 1,
            "missed_frames": 0,
            "relative_to_player": None,
        }

        output, counts = draw_detections(frame, [detection], 12.0)

        self.assertEqual(output.shape, (120, 520, 3))
        self.assertEqual(counts["slime"], 1)


if __name__ == "__main__":
    unittest.main()
