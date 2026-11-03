"""Align a locally styled image back to the deterministic structure layout.

Diffusion/ControlNet can shift panels by a few pixels.  The SVG text overlay is
drawn in the original coordinate system, so the styled image is registered to
the structure image with ORB + RANSAC before compositing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


def calibrate_styled_image(
    structure_image: str | Path,
    styled_image: str | Path,
    output_image: str | Path,
    *,
    max_features: int = 5000,
    ransac_threshold: float = 4.0,
) -> Optional[Path]:
    """Warp *styled_image* into *structure_image* coordinates.

    Returns the calibrated image path, or None when OpenCV is unavailable or
    too few reliable matches were found.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.info("OpenCV not available; skipping image calibration")
        return None

    structure = cv2.imread(str(structure_image), cv2.IMREAD_COLOR)
    styled = cv2.imread(str(styled_image), cv2.IMREAD_COLOR)
    if structure is None or styled is None:
        logger.warning("Calibration input missing")
        return None

    height, width = structure.shape[:2]
    if styled.shape[:2] != (height, width):
        styled = cv2.resize(styled, (width, height), interpolation=cv2.INTER_AREA)

    gray_structure = cv2.cvtColor(structure, cv2.COLOR_BGR2GRAY)
    gray_styled = cv2.cvtColor(styled, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp_structure, des_structure = orb.detectAndCompute(gray_structure, None)
    kp_styled, des_styled = orb.detectAndCompute(gray_styled, None)
    if des_structure is None or des_styled is None:
        logger.info("Calibration found no descriptors; using uncalibrated image")
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_styled, des_structure, k=2)
    good = [
        m
        for pair in raw_matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]
    if len(good) < 10:
        logger.info("Calibration had too few matches", matches=len(good))
        return None

    src = np.float32([kp_styled[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_structure[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
    )
    if matrix is None:
        logger.info("Calibration affine estimation failed")
        return None

    warped = cv2.warpAffine(
        styled,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    out = Path(output_image)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), warped)
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    logger.info(
        "Calibrated styled image",
        matches=len(good),
        inliers=inlier_count,
        scale=float((matrix[0, 0] ** 2 + matrix[0, 1] ** 2) ** 0.5),
        dx=float(matrix[0, 2]),
        dy=float(matrix[1, 2]),
    )
    return out


def calibrate_panels(
    structure_image: str | Path,
    styled_image: str | Path,
    output_image: str | Path,
    panel_rects: list[tuple[float, float, float, float]],
    *,
    search_margin: int = 80,
    min_score: float = 0.65,
) -> Optional[Path]:
    """Align each panel independently by template matching.

    Diffusion/ControlNet often shifts panels locally.  Matching each panel crop
    and pasting it into the expected rectangle preserves the SVG text geometry
    without needing a global homography.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.info("OpenCV not available; skipping panel calibration")
        return None

    structure = cv2.imread(str(structure_image), cv2.IMREAD_GRAYSCALE)
    styled_color = cv2.imread(str(styled_image), cv2.IMREAD_COLOR)
    if structure is None or styled_color is None:
        logger.warning("Panel calibration input missing")
        return None
    height, width = structure.shape[:2]
    styled_color = cv2.resize(styled_color, (width, height), interpolation=cv2.INTER_AREA)
    styled_gray = cv2.cvtColor(styled_color, cv2.COLOR_BGR2GRAY)
    out = styled_color.copy()
    applied = 0

    for idx, (x, y, w, h) in enumerate(panel_rects):
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            continue
        crop = structure[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        best_score = -1.0
        best: Optional[tuple[int, int, int, int]] = None
        for scale in np.arange(0.90, 1.101, 0.02):
            tpl = cv2.resize(crop, None, fx=float(scale), fy=float(scale))
            th, tw = tpl.shape[:2]
            x0 = max(0, x - search_margin)
            y0 = max(0, y - search_margin)
            x1 = min(width, x + w + search_margin)
            y1 = min(height, y + h + search_margin)
            region = styled_gray[y0:y1, x0:x1]
            if region.shape[0] < th or region.shape[1] < tw:
                continue
            result = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best = (x0 + max_loc[0], y0 + max_loc[1], tw, th)
        if best is None or best_score < min_score:
            logger.info(
                "Panel calibration confidence too low",
                panel=idx,
                score=round(best_score, 3),
                threshold=min_score,
            )
            return None
        sx, sy, sw, sh = best
        patch = cv2.resize(styled_color[sy : sy + sh, sx : sx + sw], (w, h))
        out[y : y + h, x : x + w] = patch
        applied += 1
        logger.debug("Panel calibrated", panel=idx, score=round(best_score, 3))

    if applied != len(panel_rects):
        return None

    result_path = Path(output_image)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(result_path), out)
    return result_path
