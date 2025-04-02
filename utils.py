import numpy as np
import cv2
from typing import Tuple, List

def calculate_overlap(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """Вычисление площади перекрытия между двумя контурами"""
    # Создаем маски для контуров
    mask1 = np.zeros((1000, 1000), dtype=np.uint8)
    mask2 = np.zeros((1000, 1000), dtype=np.uint8)
    
    cv2.drawContours(mask1, [contour1], -1, 255, -1)
    cv2.drawContours(mask2, [contour2], -1, 255, -1)
    
    # Находим пересечение
    intersection = cv2.bitwise_and(mask1, mask2)
    
    # Вычисляем площади
    area1 = cv2.countNonZero(mask1)
    area2 = cv2.countNonZero(mask2)
    intersection_area = cv2.countNonZero(intersection)
    
    # Возвращаем процент перекрытия
    return intersection_area / min(area1, area2)

def order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивание точек контура в порядке: верхний левый, верхний правый, нижний правый, нижний левый"""
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Сумма координат будет минимальной для верхнего левого угла
    # и максимальной для нижнего правого угла
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Разность координат будет минимальной для верхнего правого угла
    # и максимальной для нижнего левого угла
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def calculate_marker_size(corners: np.ndarray) -> float:
    """Вычисление размера маркера в пикселях"""
    # Упорядочиваем точки
    ordered = order_points(corners)
    
    # Вычисляем ширину и высоту
    width = np.linalg.norm(ordered[1] - ordered[0])
    height = np.linalg.norm(ordered[2] - ordered[1])
    
    return max(width, height)

def estimate_distance(corners: np.ndarray, marker_size: float,
                     camera_matrix: np.ndarray) -> float:
    """Оценка расстояния до маркера"""
    # Упорядочиваем точки
    ordered = order_points(corners)
    
    # Вычисляем размер маркера в пикселях
    pixel_size = calculate_marker_size(corners)
    
    # Используем формулу для оценки расстояния
    # distance = (marker_size * focal_length) / pixel_size
    focal_length = camera_matrix[0, 0]
    distance = (marker_size * focal_length) / pixel_size
    
    return distance 