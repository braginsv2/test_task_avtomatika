import cv2
import numpy as np
import math

def detect_marker(image_path):
    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")
    
    # Сохраняем копию исходного изображения
    original = img.copy()
    
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Бинаризуем изображение (инвертируем, чтобы белое стало черным)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Находим самый большой контур (предполагаем, что это наш маркер)
    if not contours:
        raise ValueError("Маркер не найден")
    
    max_contour = max(contours, key=cv2.contourArea)
    
    # Получаем минимальный ограничивающий прямоугольник
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # Создаем изображение только с контуром
    contour_img = original.copy()
    cv2.drawContours(contour_img, [max_contour], -1, (0, 255, 0), 2)
    
    # Рисуем прямоугольник на основном изображении
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    
    # Получаем угол поворота
    angle = abs(rect[2]-90)
    
    # Получаем площадь прямоугольника
    area = rect[1][0] * rect[1][1]

    # Вычисляем дистанцию
    distance = 0.15 * math.sqrt(89401.0/area)

    
    return original, contour_img, img, angle, area, distance 