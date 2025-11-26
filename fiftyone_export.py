"""
Utilities to export detected faces (crops + embeddings) for visualization in FiftyOne.

Copy/paste this into a notebook cell (after the existing detection helpers are defined)
or import the module directly in the notebook:

    !pip install -q fiftyone
    from fiftyone_export import export_faces_for_fiftyone
    dataset, session = export_faces_for_fiftyone(
        video_path=f"/content/drive/MyDrive/HeadCountStuff/video-data/{filenames[1]}",
        process_fps=1.0,
        use_cpu=False,
    )

Prerequisites expected in the notebook session:
- `detect_faces_and_embeddings` and `load_retinaface_detector` already defined (from the provided code).
- OpenCV (`cv2`) installed and importable.
"""

import os
from typing import List, Optional, Tuple

import cv2
import fiftyone as fo
import numpy as np


def _ensure_frame_interval(cap: cv2.VideoCapture, process_fps: float) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    return max(int(round(fps / process_fps)), 1)


def export_faces_for_fiftyone(
    video_path: str,
    output_dir: str = "/content/drive/MyDrive/HeadCountStuff/face-crops",
    process_fps: float = 1.0,
    use_cpu: bool = False,
    min_size: int = 0,
    max_offset_ratio: float = 0.80,
    min_conf: float = 0.5,
    dataset_name: str = "headcount_faces",
    face_detector=None,
) -> Tuple[fo.Dataset, "fo.session.session"]:
    """
    Runs face detection at a reduced FPS, saves cropped faces to disk, and builds
    a FiftyOne dataset with embeddings for interactive visualization.

    Args:
        video_path: Path to the input video.
        output_dir: Directory where cropped face images will be written.
        process_fps: How many frames per second to process (sampling rate).
        use_cpu: Whether to run inference on CPU (passed to load_retinaface_detector).
        min_size: Minimum face size (pixels) to keep.
        max_offset_ratio: Frontal filter threshold from detect_faces_and_embeddings.
        min_conf: Minimum detection confidence to keep.
        dataset_name: Name of the FiftyOne dataset (overwrites if it exists).
        face_detector: Optional preloaded detector from load_retinaface_detector.

    Returns:
        (dataset, session) where dataset is the FiftyOne Dataset and session is
        the launched FiftyOne App session.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    if face_detector is None:
        face_detector = load_retinaface_detector(use_cpu=use_cpu)

    frame_interval = _ensure_frame_interval(cap, process_fps)

    samples: List[fo.Sample] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        faces = detect_faces_and_embeddings(
            frame,
            face_detector,
            min_size=min_size,
            max_offset_ratio=max_offset_ratio,
            min_conf=min_conf,
        )

        for j, face in enumerate(faces):
            x, y, w, h = face["box"]
            crop = frame[y : y + h, x : x + w]

            crop_path = os.path.join(output_dir, f"f{frame_idx:06d}_p{j}.jpg")
            cv2.imwrite(crop_path, crop)

            sample = fo.Sample(filepath=crop_path)
            sample["frame_index"] = frame_idx
            sample["detection_score"] = float(face["score"])
            # Store embedding as a vector field for FiftyOne visualizations
            sample["embedding"] = fo.Vector(np.asarray(face["embedding"], dtype=float).tolist())
            samples.append(sample)

        frame_idx += 1

    cap.release()

    dataset = fo.Dataset(dataset_name, overwrite=True)
    dataset.add_samples(samples)

    print(f"Saved {len(samples)} face crops to {output_dir}")
    print(f"Dataset '{dataset.name}' ready for visualization in FiftyOne.")
    session = fo.launch_app(dataset)
    return dataset, session


if __name__ == "__main__":
    # Minimal CLI usage example (adjust the path to your video):
    # python fiftyone_export.py "/path/to/video.mp4"
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fiftyone_export.py <video_path>")
        sys.exit(1)

    export_faces_for_fiftyone(sys.argv[1])
