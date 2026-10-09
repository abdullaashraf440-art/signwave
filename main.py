#!/usr/bin/env python3
"""
main.py
========

Real-time dual-modal (vision + sequence/NLP) sign language translation
pipeline.

    webcam frame
        -> vision.LandmarkExtractor        (spatial feature extraction)
        -> vision.preprocessing            (normalize + window)
        -> nlp.SignLanguageTransformer     (sequence modeling, CTC logits)
        -> nlp.CTCGreedyDecoder            (frame-level gesture alignment -> glosses)
        -> nlp.GlossToTextRefiner          (glosses -> fluent sentence, HF transformer)
        -> nlp.TextToSpeech                (sentence -> spoken audio)

Run:
    python main.py --config configs/config.yaml
    python main.py --no-display --max-frames 300   # headless smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

from vision import (
    LandmarkExtractor,
    VideoStream,
    build_sequence_window,
    flatten_frame_landmarks,
    normalize_landmarks,
)
from vision.preprocessing import frame_activity_score
from nlp import CTCGreedyDecoder, GlossToTextRefiner, SignLanguageTransformer, TextToSpeech

logger = logging.getLogger("sign_language_translator")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_vocab(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        logger.warning("Vocab file %s not found; using a tiny placeholder vocab.", path)
        return ["hello", "thank_you", "yes", "no"]
    with open(p, "r") as f:
        return json.load(f)


def build_model(cfg: dict, vocab_size: int) -> SignLanguageTransformer:
    sm_cfg = cfg["sequence_model"]
    model = SignLanguageTransformer(
        input_dim=sm_cfg["input_dim"],
        d_model=sm_cfg["d_model"],
        nhead=sm_cfg["nhead"],
        num_layers=sm_cfg["num_layers"],
        dim_feedforward=sm_cfg["dim_feedforward"],
        vocab_size=vocab_size,
    )
    ckpt_path = Path(sm_cfg.get("checkpoint_path", ""))
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from %s", ckpt_path)
    else:
        logger.warning(
            "No checkpoint found at %s -- running with randomly initialized "
            "weights (pipeline-wiring / latency demo mode only).",
            ckpt_path,
        )
    model.eval()
    return model


class GlossStabilizer:
    """Only "commits" a decoded gloss sequence once it has been stable
    for `stability_window` consecutive model passes, so the pipeline
    doesn't spam half-formed sentences to text/TTS while a gesture is
    still in progress.
    """

    def __init__(self, stability_window: int = 6) -> None:
        self.stability_window = stability_window
        self._history: deque[list[str]] = deque(maxlen=stability_window)
        self._last_committed: list[str] | None = None

    def update(self, glosses: list[str]) -> Optional[list[str]]:
        self._history.append(glosses)
        if len(self._history) < self.stability_window:
            return None
        if all(h == glosses for h in self._history) and glosses and glosses != self._last_committed:
            self._last_committed = glosses
            return glosses
        return None


def run(cfg: dict, display: bool, max_frames: Optional[int] = None) -> None:
    logging.basicConfig(level=cfg["runtime"].get("log_level", "INFO"))

    vocab = load_vocab(cfg["sequence_model"]["vocab_path"])
    model = build_model(cfg, vocab_size=len(vocab))
    decoder = CTCGreedyDecoder(vocab=vocab, blank_id=model.blank_id)
    refiner = GlossToTextRefiner(
        model_name=cfg["nlp"]["hf_model_name"], use_hf=cfg["nlp"]["refine_with_hf"]
    )
    tts = TextToSpeech(backend=cfg["tts"]["backend"], rate=cfg["tts"]["rate"], voice=cfg["tts"]["voice"])
    stabilizer = GlossStabilizer(cfg["nlp"]["gloss_stability_window"])

    window_size = cfg["sequence_model"]["window_size"]
    stride = cfg["sequence_model"]["inference_stride"]
    activity_threshold = cfg["vision"]["activity_threshold"]

    frame_buffer: deque[np.ndarray] = deque(maxlen=window_size)
    current_text = ""
    frame_count = 0
    fps_timer = time.time()
    fps_frames = 0
    fps_display = 0.0

    cam_cfg = cfg["camera"]
    with VideoStream(
        source=cam_cfg["source"], width=cam_cfg["width"], height=cam_cfg["height"], fps=cam_cfg["fps"]
    ) as stream, LandmarkExtractor(
        model_complexity=cfg["vision"]["model_complexity"],
        min_detection_confidence=cfg["vision"]["min_detection_confidence"],
        min_tracking_confidence=cfg["vision"]["min_tracking_confidence"],
    ) as extractor:
        logger.info("Pipeline started. Press 'q' in the video window to quit.")
        try:
            while True:
                ok, frame = stream.read()
                if not ok or frame is None:
                    continue

                landmarks = extractor.process(frame)
                pose_n, left_n, right_n = normalize_landmarks(
                    landmarks.pose, landmarks.left_hand, landmarks.right_hand
                )
                feature_vec = flatten_frame_landmarks(pose_n, left_n, right_n)
                frame_buffer.append(feature_vec)

                activity = frame_activity_score(
                    landmarks.left_hand,
                    landmarks.right_hand,
                    landmarks.left_hand_detected,
                    landmarks.right_hand_detected,
                )

                if frame_count % stride == 0 and activity >= activity_threshold:
                    window = build_sequence_window(list(frame_buffer), window_size)
                    log_probs = model.infer(torch.from_numpy(window))
                    glosses = decoder.decode(log_probs)

                    committed = stabilizer.update(glosses)
                    if committed:
                        sentence = refiner.refine(committed)
                        if sentence:
                            current_text = sentence
                            logger.info("Decoded: %s", sentence)
                            tts.speak(sentence)

                if display:
                    annotated = extractor.draw(frame) if cfg["runtime"]["overlay_landmarks"] else frame
                    fps_frames += 1
                    if time.time() - fps_timer >= 1.0:
                        fps_display = fps_frames / (time.time() - fps_timer)
                        fps_frames = 0
                        fps_timer = time.time()

                    cv2.putText(
                        annotated, current_text, (10, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated, f"{fps_display:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
                    )
                    cv2.imshow("Sign Language Translator", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            tts.stop()
            if display:
                cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no cv2 window)")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Stop after N frames (useful for smoke tests/CI)"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Override video source from config: a webcam index (e.g. 0) or a video file path",
    )
    return parser.parse_args()


def _coerce_source(value: str) -> "int | str":
    """--camera accepts either a webcam index or a video file path; a
    plain integer string means "webcam index", anything else (e.g. a
    path, used for offline smoke tests) is passed through as-is.
    """
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera"]["source"] = _coerce_source(args.camera)
    display = cfg["runtime"].get("display", True) and not args.no_display
    run(cfg, display=display, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
