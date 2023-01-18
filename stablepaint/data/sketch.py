from __future__ import annotations

import cv2
import numpy as np


def dog(x: np.ndarray, size: tuple[int, int], sigma: float, k: float, gamma: float) -> np.ndarray:
    if x.shape[-1] == 3: x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(x, size, sigma) - gamma * cv2.GaussianBlur(x, size, sigma * k)


def xdog(x: np.ndarray, sigma: float, k: float, gamma: float, epsilon: float, phi: float) -> np.ndarray:
    x = dog(x, (0, 0), sigma, k, gamma) / 255
    return np.where(x < epsilon, 255, 255 * (1 + np.tanh(phi * x))).astype(np.uint8)


def sketch(x: np.ndarray, sigma: float, k: float, gamma: float, epsilon: float, phi: float, area_min: int) -> np.ndarray:
    x = xdog(x, sigma, k, gamma, epsilon, phi)
    x = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = [contour for contour in contours if cv2.contourArea(contour) > area_min]
    return 255 - cv2.drawContours(np.zeros_like(x), contours, -1, (255, 255, 255), -1)