import cv2
import numpy as np
import argparse
from typing import Tuple, Optional

class MarkerDetector:
    def __init__(self):
        self.min_marker_size = 100  # минимальный размер маркера
        self.max_marker_size = 500000  # максимальный размер маркера
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предварительная обработка изображения"""
        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Нормализация контраста
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Применяем размытие по Гауссу для уменьшения шума
        blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
        
        # Применяем адаптивную пороговую обработку с меньшим размером окна
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 7, 2)
        
        # Морфологические операции для улучшения качества
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Дополнительная морфологическая обработка для удаления шума
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        return thresh
    
    def find_marker_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Поиск углов маркера"""
        # Пробуем все методы поиска с разными параметрами
        methods = [
            (self.find_marker_by_contours, {}),
            (self.find_marker_by_corners, {}),
            (self.find_marker_by_lines, {}),
            # Повторяем методы с более мягкими параметрами
            (self.find_marker_by_contours, {'min_size_factor': 0.3, 'angle_threshold': 50}),
            (self.find_marker_by_corners, {'angle_threshold': 55}),
            (self.find_marker_by_lines, {'angle_threshold': 60})
        ]
        
        for method, params in methods:
            marker = method(image, **params)
            if marker is not None:
                return marker
        
        return None
    
    def find_marker_by_contours(self, image: np.ndarray, min_size_factor: float = 1.0,
                              angle_threshold: float = 45) -> Optional[np.ndarray]:
        """Поиск маркера через контуры"""
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        min_size = self.min_marker_size * min_size_factor
        
        for contour in contours[:5]:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if min_size < area < self.max_marker_size:
                    angles = self.check_angles(approx)
                    if all(abs(angle - 90) < angle_threshold for angle in angles):
                        if self.check_aspect_ratio(approx):
                            return approx.reshape(4, 1, 2)
        return None
    
    def find_marker_by_corners(self, image: np.ndarray,
                             angle_threshold: float = 50) -> Optional[np.ndarray]:
        """Поиск маркера через углы Харриса"""
        corners = cv2.cornerHarris(image, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        corner_points = []
        threshold = 0.01 * corners.max()
        for i in range(corners.shape[0]):
            for j in range(corners.shape[1]):
                if corners[i,j] > threshold:
                    corner_points.append([j, i])
        
        if len(corner_points) < 4:
            return None
            
        corner_points = np.array(corner_points, dtype=np.float32)
        hull = cv2.convexHull(corner_points)
        
        if len(hull) < 4:
            return None
            
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if self.min_marker_size < area < self.max_marker_size:
                angles = self.check_angles(approx)
                if all(abs(angle - 90) < angle_threshold for angle in angles):
                    if self.check_aspect_ratio(approx):
                        return approx.reshape(4, 1, 2)
        return None
    
    def find_marker_by_lines(self, image: np.ndarray,
                           angle_threshold: float = 55) -> Optional[np.ndarray]:
        """Поиск маркера через линии Хафа"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None or len(lines) < 4:
            return None
            
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]
                
                if abs(theta1 - theta2) > 0.1:
                    A = np.array([
                        [np.cos(theta1), np.sin(theta1)],
                        [np.cos(theta2), np.sin(theta2)]
                    ])
                    b = np.array([rho1, rho2])
                    
                    try:
                        x, y = np.linalg.solve(A, b)
                        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                            intersections.append([x, y])
                    except:
                        continue
        
        if len(intersections) < 4:
            return None
            
        intersections = np.array(intersections, dtype=np.float32)
        hull = cv2.convexHull(intersections)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if self.min_marker_size < area < self.max_marker_size:
                angles = self.check_angles(approx)
                if all(abs(angle - 90) < angle_threshold for angle in angles):
                    if self.check_aspect_ratio(approx):
                        return approx.reshape(4, 1, 2)
        return None
    
    def find_complete_marker(self, contours: list) -> Optional[np.ndarray]:
        """Поиск целого маркера"""
        for contour in contours[:3]:
            # Аппроксимируем контур
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Проверяем, что контур имеет 4 угла (маркер)
            if len(approx) == 4:
                # Проверяем размер маркера
                area = cv2.contourArea(contour)
                if self.min_marker_size < area < self.max_marker_size:
                    # Проверяем, что углы примерно прямые
                    angles = self.check_angles(approx)
                    if all(abs(angle - 90) < 45 for angle in angles):
                        # Проверяем соотношение сторон
                        if self.check_aspect_ratio(approx):
                            return approx
        return None
    
    def find_partial_marker(self, contours: list) -> Optional[np.ndarray]:
        """Поиск частей маркера и восстановление его формы"""
        for i, contour in enumerate(contours[:10]):  # проверяем больше контуров
            # Аппроксимируем контур
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Ищем части маркера (3 или 4 угла)
            if len(approx) in [3, 4]:
                area = cv2.contourArea(contour)
                if area > self.min_marker_size * 0.3:  # уменьшаем минимальный размер
                    # Проверяем углы
                    angles = self.check_angles(approx)
                    if all(abs(angle - 90) < 50 for angle in angles):  # увеличиваем допуск
                        # Пытаемся восстановить маркер
                        marker = self.reconstruct_marker(approx)
                        if marker is not None:
                            return marker
        return None
    
    def reconstruct_marker(self, partial_corners: np.ndarray) -> Optional[np.ndarray]:
        """Восстановление полного маркера из его части"""
        if len(partial_corners) == 4:
            return partial_corners
            
        # Если у нас 3 угла, пытаемся восстановить четвертый
        if len(partial_corners) == 3:
            # Упорядочиваем точки
            ordered = self.order_points(partial_corners)
            
            # Вычисляем предполагаемое положение четвертого угла
            p1, p2, p3 = ordered
            v1 = p2 - p1
            v2 = p3 - p2
            p4 = p1 + v2  # предполагаем, что маркер прямоугольный
            
            # Добавляем четвертый угол
            corners = np.vstack((ordered, p4))
            
            # Проверяем получившийся маркер
            if self.check_aspect_ratio(corners):
                return corners
                
        return None
    
    def check_corner_markers(self, image: np.ndarray, corners: np.ndarray) -> bool:
        """Проверка наличия угловых маркеров"""
        # Упорядочиваем точки
        ordered = self.order_points(corners)
        
        # Размер области для проверки угловых маркеров
        check_size = 60  # увеличиваем размер области проверки
        
        # Проверяем каждый угол
        valid_corners = 0
        for i in range(4):
            # Получаем координаты угла
            x, y = ordered[i]
            
            # Проверяем область вокруг угла
            roi = image[int(y-check_size):int(y+check_size), 
                       int(x-check_size):int(x+check_size)]
            
            if roi.size == 0:
                continue
                
            # Проверяем, что в области есть черный пиксель
            if np.min(roi) < 127:  # если есть хотя бы один темный пиксель
                valid_corners += 1
        
        # Требуем как минимум 2 валидных угла
        return valid_corners >= 2
    
    def check_angles(self, corners: np.ndarray) -> list:
        """Проверка углов маркера"""
        angles = []
        for i in range(4):
            pt1 = corners[i][0]
            pt2 = corners[(i + 1) % 4][0]
            pt3 = corners[(i + 2) % 4][0]
            
            # Вычисляем векторы
            v1 = pt1 - pt2
            v2 = pt3 - pt2
            
            # Вычисляем угол
            angle = np.abs(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)) * 180 / np.pi)
            angles.append(angle)
        
        return angles
    
    def check_aspect_ratio(self, corners: np.ndarray) -> bool:
        """Проверка соотношения сторон маркера"""
        # Упорядочиваем точки
        ordered = self.order_points(corners)
        
        # Вычисляем длины сторон
        sides = []
        for i in range(4):
            pt1 = ordered[i]
            pt2 = ordered[(i + 1) % 4]
            length = np.linalg.norm(pt2 - pt1)
            sides.append(length)
        
        # Проверяем, что все стороны примерно равны
        max_diff = max(sides) - min(sides)
        avg_length = sum(sides) / 4
        return max_diff / avg_length < 0.7  # уменьшаем допуск на соотношение сторон
    
    def estimate_pose(self, corners: np.ndarray, camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Оценка положения маркера"""
        # Размер маркера в метрах
        marker_size = 0.1
        
        # Упорядочиваем точки
        corners = self.order_points(corners)
        
        # Точки маркера в 3D
        obj_points = np.array([
            [0, 0, 0],
            [marker_size, 0, 0],
            [marker_size, marker_size, 0],
            [0, marker_size, 0]
        ], dtype=np.float32)
        
        # Решаем PnP задачу
        ret, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)
        
        return rvec, tvec
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Упорядочивание точек контура"""
        # Преобразуем точки в формат (4, 2)
        pts = pts.reshape(4, 2)
        
        # Создаем массив для упорядоченных точек
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
    
    def detect_marker(self, image: np.ndarray, camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray) -> Optional[dict]:
        """Основной метод обнаружения маркера"""
        # Предварительная обработка
        processed = self.preprocess_image(image)
        
        # Поиск углов
        corners = self.find_marker_corners(processed)
        if corners is None:
            return None
            
        # Убеждаемся, что углы в правильном формате для оценки положения
        corners_pose = corners.reshape(4, 2)
            
        # Оценка положения
        rvec, tvec = self.estimate_pose(corners_pose, camera_matrix, dist_coeffs)
        
        # Преобразуем углы Эйлера в градусы
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        
        # Убеждаемся, что углы в правильном формате для отрисовки
        corners_draw = corners.reshape(4, 1, 2).astype(np.int32)
        
        return {
            'corners': corners_draw,
            'rotation': euler_angles,
            'translation': tvec
        }

def main():
    parser = argparse.ArgumentParser(description='Детектор визуальных меток')
    parser.add_argument('--camera', type=int, default=0, help='Номер камеры')
    args = parser.parse_args()
    
    # Инициализация камеры
    cap = cv2.VideoCapture(args.camera)
    
    # Примерные параметры камеры (нужно откалибровать)
    camera_matrix = np.array([[1000, 0, 320],
                            [0, 1000, 240],
                            [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    detector = MarkerDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Обнаружение маркера
        result = detector.detect_marker(frame, camera_matrix, dist_coeffs)
        
        if result is not None:
            # Отрисовка результатов
            cv2.drawContours(frame, [result['corners']], -1, (0, 255, 0), 2)
            
            # Вывод информации
            text = f"Rotation: {result['rotation']:.1f}°, Distance: {result['translation'][2].item():.2f}m"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Показываем обработанное изображение
        cv2.imshow('Processed', processed)
        cv2.imshow('Marker Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 