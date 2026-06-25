"""ChokePoint gallery visibility metrics.

For this evaluation, a gallery face is considered in-scope when it appears in
the ChokePoint ground-truth labels. We keep recognition_eligible on the record
for compatibility with older logs, but these metrics do not filter on it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GalleryFaceObs:
    frame_id: int
    person_id: str
    recognition_eligible: bool
    allowed_raw: bool


def real_enrolled_gallery_allow_rate(observations: list[GalleryFaceObs]) -> float:
    if not observations:
        return 0.0
    allowed = sum(1 for o in observations if o.allowed_raw)
    return allowed / len(observations)


def time_to_first_allow(
    observations: list[GalleryFaceObs],
    fps: float = 5.0,
) -> dict:
    """Return mean time-to-first-allow in frames and milliseconds.

    fps: source (original) video FPS used to convert frame diffs to ms.
         NOT the pipeline target FPS.
    """
    obs_by_person: dict[str, list[GalleryFaceObs]] = {}
    for o in observations:
        obs_by_person.setdefault(o.person_id, []).append(o)
    for pid in obs_by_person:
        obs_by_person[pid].sort(key=lambda o: o.frame_id)

    ttfa_frames_list: list[float] = []
    ttfa_ms_list: list[float] = []

    for _pid, person_obs in obs_by_person.items():
        first_gallery_frame = person_obs[0].frame_id
        allowed_obs = [o for o in person_obs if o.allowed_raw]
        if not allowed_obs:
            continue
        first_allow_frame = allowed_obs[0].frame_id
        frames_diff = first_allow_frame - first_gallery_frame
        ttfa_frames_list.append(float(frames_diff))
        if fps > 0:
            ttfa_ms_list.append(frames_diff * 1000.0 / fps)

    if not ttfa_frames_list:
        return {"ttfa_frames": None, "ttfa_ms": None}

    return {
        "ttfa_frames": sum(ttfa_frames_list) / len(ttfa_frames_list),
        "ttfa_ms": sum(ttfa_ms_list) / len(ttfa_ms_list) if ttfa_ms_list else None,
    }


def allow_stability(observations: list[GalleryFaceObs]) -> float:
    obs_by_person: dict[str, list[GalleryFaceObs]] = {}
    for o in observations:
        obs_by_person.setdefault(o.person_id, []).append(o)
    for pid in obs_by_person:
        obs_by_person[pid].sort(key=lambda o: o.frame_id)

    total_after_first = 0
    allowed_after_first = 0

    for _pid, person_obs in obs_by_person.items():
        allowed_obs = [o for o in person_obs if o.allowed_raw]
        if not allowed_obs:
            continue
        first_allow_frame = allowed_obs[0].frame_id
        after = [o for o in person_obs if o.frame_id >= first_allow_frame]
        total_after_first += len(after)
        allowed_after_first += sum(1 for o in after if o.allowed_raw)

    if total_after_first == 0:
        return 0.0
    return allowed_after_first / total_after_first
