from scripts.eval_mot20_detection_overlap import (
    Box,
    contains,
    evaluate_body_roi_coverage,
    evaluate_face_detections,
    evaluate_person_detections,
    greedy_match_iou,
    iou,
    make_stretch_mapper,
    map_fps_frame,
    overlaps,
)


def test_iou_calculation():
    assert iou(Box(1, 0, 0, 10, 10), Box(1, 5, 5, 10, 10)) == 25 / 175
    assert iou(Box(1, 0, 0, 10, 10), Box(1, 20, 20, 10, 10)) == 0.0


def test_greedy_iou_matching_is_one_to_one():
    preds = [
        Box(1, 0, 0, 10, 10),
        Box(1, 1, 1, 10, 10),
    ]
    refs = [Box(1, 0, 0, 10, 10)]
    matches, matched_preds, matched_refs = greedy_match_iou(preds, refs, 0.5)

    assert len(matches) == 1
    assert matched_preds == 1
    assert matched_refs == 1


def test_full_face_roi_containment():
    person = Box(1, 10, 10, 100, 200)
    assert contains(person, Box(1, 20, 20, 10, 10))
    assert not contains(person, Box(1, 5, 20, 10, 10))
    assert not contains(person, Box(1, 20, 20, 100, 10))


def test_body_roi_overlap():
    public_body = Box(1, 10, 10, 100, 200)
    assert overlaps(public_body, Box(1, 0, 0, 20, 20))
    assert not overlaps(public_body, Box(1, 200, 200, 10, 10))


def test_stretch_frame_mapping():
    mapper = make_stretch_mapper(max_friend_frame=2781, ref_min=1, ref_max=429)
    assert mapper(0) == 1
    assert mapper(2781) == 429
    assert 214 <= mapper(1390) <= 215


def test_fps_frame_mapping():
    assert map_fps_frame(0, 25.0, 5.0, 1) == 1
    assert map_fps_frame(5, 25.0, 5.0, 1) == 2
    assert map_fps_frame(30, 30.0, 5.0, 1) == 6


def test_person_eval_counts_tp_fp_fn():
    refs = {1: [Box(1, 0, 0, 10, 10), Box(1, 100, 100, 10, 10)]}
    preds = {1: [Box(1, 0, 0, 10, 10), Box(1, 200, 200, 10, 10)]}
    summary, _ = evaluate_person_detections(preds, refs, threshold=0.5)

    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5


def test_body_roi_coverage_counts_intersections_and_covered_boxes():
    refs = {1: [Box(1, 0, 0, 100, 100), Box(1, 200, 200, 100, 100)]}
    rois = {1: [Box(1, 50, 50, 10, 10), Box(1, 500, 500, 10, 10)]}
    summary, _ = evaluate_body_roi_coverage(rois, refs)

    assert summary["total_rois"] == 2
    assert summary["evaluated_rois"] == 2
    assert summary["rois_intersecting_public_box"] == 1
    assert summary["rois_not_intersecting_public_box"] == 1
    assert summary["covered_public_boxes"] == 1


def test_face_eval_out_of_reference_and_duplicate_coverage():
    refs = {1: [Box(1, 0, 0, 100, 100)], 2: [Box(2, 0, 0, 100, 100)]}
    faces = [
        Box(0, 10, 10, 10, 10),
        Box(0, 20, 20, 10, 10),
        Box(1, 200, 200, 10, 10),
        Box(99, 10, 10, 10, 10),
    ]

    summary, _ = evaluate_face_detections(
        faces,
        refs,
        mapper=lambda frame: frame + 1,
        ref_min=1,
        ref_max=2,
    )

    assert summary["total_faces"] == 4
    assert summary["evaluated_faces"] == 3
    assert summary["out_of_reference_faces"] == 1
    assert summary["faces_intersecting_public_box"] == 2
    assert summary["faces_not_intersecting_public_box"] == 1
    assert summary["covered_public_boxes"] == 1
