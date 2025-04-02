import cv2
import numpy as np
from marker_generator import MarkerGenerator
from marker_detector import MarkerDetector
import time

def test_marker_generation():
    """Тест генерации маркера"""
    print("\nТест генерации маркера:")
    generator = MarkerGenerator()
    marker = generator.generate_marker(marker_id=1)
    cv2.imwrite('test_marker.png', marker)
    
    # Показываем сгенерированный маркер
    cv2.imshow('Generated Marker', marker)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Маркер успешно сгенерирован")

def test_marker_detection():
    """Тест обнаружения маркера"""
    print("\nТест обнаружения маркера:")
    # Загружаем маркер
    marker = cv2.imread('test_marker.png')
    if marker is None:
        print("Ошибка: не удалось загрузить маркер")
        return
    
    # Показываем исходный маркер
    cv2.imshow('Original Marker', marker)
    cv2.waitKey(1000)
    
    # Создаем детектор
    detector = MarkerDetector()
    
    # Получаем обработанное изображение
    processed = detector.preprocess_image(marker)
    cv2.imshow('Processed Image', processed)
    cv2.waitKey(1000)
    
    # Примерные параметры камеры (нужно откалибровать)
    camera_matrix = np.array([[1000, 0, 320],
                            [0, 1000, 240],
                            [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    # Обнаружение маркера
    result = detector.detect_marker(marker, camera_matrix, dist_coeffs)
    
    if result is not None:
        print("Маркер успешно обнаружен")
        print(f"Углы: {result['rotation']}")
        print(f"Расстояние: {result['translation'][2].item():.2f}м")
        
        # Отображаем результат
        result_image = marker.copy()
        cv2.drawContours(result_image, [result['corners']], -1, (0, 255, 0), 2)
        cv2.imshow('Detection Result', result_image)
        cv2.waitKey(1000)
    else:
        print("Маркер не обнаружен")
    
    cv2.destroyAllWindows()

def test_overlap_resistance():
    """Тест устойчивости к перекрытию"""
    print("\nТест устойчивости к перекрытию:")
    # Загружаем маркер
    marker = cv2.imread('test_marker.png')
    if marker is None:
        print("Ошибка: не удалось загрузить маркер")
        return
    
    # Создаем перекрытую версию маркера
    height, width = marker.shape[:2]
    overlap = int(width * 0.3)  # 30% перекрытие
    overlapped = marker.copy()
    overlapped[:, -overlap:] = 255  # Белый прямоугольник справа
    
    # Сохраняем перекрытый маркер
    cv2.imwrite('test_marker_overlap.png', overlapped)
    
    # Создаем детектор
    detector = MarkerDetector()
    
    # Примерные параметры камеры
    camera_matrix = np.array([[1000, 0, 320],
                            [0, 1000, 240],
                            [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    # Обнаружение маркера
    result = detector.detect_marker(overlapped, camera_matrix, dist_coeffs)
    
    if result is not None:
        print("Маркер успешно обнаружен с перекрытием 30%")
        print(f"Углы: {result['rotation']}")
        print(f"Расстояние: {result['translation'][2].item():.2f}м")
        
        # Отображаем результат
        result_image = overlapped.copy()
        cv2.drawContours(result_image, [result['corners']], -1, (0, 255, 0), 2)
        cv2.imshow('Overlap Detection Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Маркер не обнаружен с перекрытием")
        # Показываем обработанное изображение для анализа
        processed = detector.preprocess_image(overlapped)
        cv2.imshow('Processed Overlapped Image', processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    print("Запуск тестов...")
    
    # Тест генерации маркера
    test_marker_generation()
    
    # Тест обнаружения маркера
    test_marker_detection()
    
    # Тест устойчивости к перекрытию
    test_overlap_resistance()
    
    print("\nТесты завершены")

if __name__ == '__main__':
    main() 