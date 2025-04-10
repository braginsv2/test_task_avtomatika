import cv2
import numpy as np
import time
from marker_generator import MarkerGenerator
from marker_detector import detect_marker

def test_detector():
    # Создаем генератор маркеров
    generator = MarkerGenerator()
    
    # Тестируем маркеры с разными ID
    for marker_id in [1, 2, 3]:
        print(f"\nТестирование маркера с ID {marker_id}")
        
        # Генерируем маркер
        test_marker = generator.generate_marker(marker_id)
        filename = f'marker_{marker_id}.png'
        cv2.imwrite(filename, test_marker)
        
        # Детектируем маркер
        try:
            original, contour_img, result_img, angle, area, distance = detect_marker(filename)
            
            # Выводим результаты
            print(f"ID маркера: {marker_id}")
            print(f"Угол поворота: {angle:.1f} градусов")
            print(f"Дистанция: {distance:.3f} метров")
            
            # Показываем исходный маркер
            cv2.imshow(f'original_marker {marker_id}', original)
            cv2.waitKey(1000)  # Ждем 1 секунду
            
            # Показываем контур
            cv2.imshow(f'contour_marker {marker_id}', contour_img)
            cv2.waitKey(1000)  # Ждем 1 секунду
            
            # Показываем прямоугольник
            cv2.imshow(f'detected_marker {marker_id}', result_img)
            cv2.waitKey(1000)  # Ждем 1 секунду
            
            # Закрываем все окна
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Ошибка при детектировании маркера {marker_id}: {str(e)}")
            cv2.destroyAllWindows()

def test_size_variations():
    print("\nТестирование маркера с разными размерами")
    generator = MarkerGenerator()
    
    # Генерируем базовый маркер с ID 1
    base_marker = generator.generate_marker(1)
    
    # Увеличиваем маркер в 1.5 раза
    enlarged_marker = cv2.resize(base_marker, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('enlarged_marker.png', enlarged_marker)
    
    # Уменьшаем маркер в 2 раза
    reduced_marker = cv2.resize(base_marker, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('reduced_marker.png', reduced_marker)
    
    # Тестируем увеличенный маркер
    print("\nТестирование увеличенного маркера (1.5x)")
    try:
        original, contour_img, result_img, angle, area, distance = detect_marker('enlarged_marker.png')
        print(f"ID маркера: 1")
        print(f"Угол поворота: {angle:.1f} градусов")
        print(f"Дистанция: {distance:.3f} метров")
        
        cv2.imshow('enlarged_marker', original)
        cv2.waitKey(1000)
        cv2.imshow('contour_enlarged_marker', contour_img)
        cv2.waitKey(1000)
        cv2.imshow('detected_enlarged_marker', result_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка при детектировании увеличенного маркера: {str(e)}")
        cv2.destroyAllWindows()
    
    # Тестируем уменьшенный маркер
    print("\nТестирование уменьшенного маркера (0.5x)")
    try:
        original, contour_img, result_img, angle, area, distance = detect_marker('reduced_marker.png')
        print(f"ID маркера: 1")
        print(f"Угол поворота: {angle:.1f} градусов")
        print(f"Дистанция: {distance:.3f} метров")
        
        cv2.imshow('reduced_marker', original)
        cv2.waitKey(1000)
        cv2.imshow('contour_reduced_marker', contour_img)
        cv2.waitKey(1000)
        cv2.imshow('detected_reduced_marker', result_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка при детектировании уменьшенного маркера: {str(e)}")
        cv2.destroyAllWindows()

def test_rotation():
    print("\nТестирование маркера с разными углами поворота")
    generator = MarkerGenerator()
    
    # Генерируем базовый маркер с ID 1
    base_marker = generator.generate_marker(1)
    
    # Тестируем разные углы поворота
    for angle in [15, 30, 45]:
        print(f"\nТестирование маркера с поворотом на {angle} градусов")
        
        # Получаем размеры изображения
        h, w = base_marker.shape[:2]
        
        # Вычисляем центр изображения
        center = (w // 2, h // 2)
        
        # Создаем матрицу поворота
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Применяем поворот
        rotated_marker = cv2.warpAffine(base_marker, rotation_matrix, (w, h))
        
        # Сохраняем повернутый маркер
        filename = f'rotated_marker_{angle}.png'
        cv2.imwrite(filename, rotated_marker)
        
        # Детектируем маркер
        try:
            original, contour_img, result_img, detected_angle, area, distance = detect_marker(filename)
            
            # Выводим результаты
            print(f"ID маркера: 1")
            print(f"Обнаруженный угол поворота: {detected_angle:.1f} градусов")
            print(f"Дистанция: {distance:.3f} метров")
            
            # Показываем повернутый маркер
            cv2.imshow(f'rotated_marker {angle}', original)
            cv2.waitKey(1000)
            
            # Показываем контур
            cv2.imshow(f'contour_rotated_marker {angle}', contour_img)
            cv2.waitKey(1000)
            
            # Показываем прямоугольник
            cv2.imshow(f'detected_rotated_marker {angle}', result_img)
            cv2.waitKey(1000)
            
            # Закрываем все окна
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Ошибка при детектировании повернутого маркера: {str(e)}")
            cv2.destroyAllWindows()

def test_noise():
    print("\nТестирование маркера с шумом")
    generator = MarkerGenerator()
    
    # Генерируем базовый маркер с ID 1
    base_marker = generator.generate_marker(1)
    
    # Создаем копию маркера для добавления шума
    noisy_marker = base_marker.copy()
    
    # Получаем размеры изображения
    h, w = noisy_marker.shape[:2]
    
    # Вычисляем количество пикселей для зашумления (30%)
    num_pixels = int(h * w * 0.3)
    
    # Создаем маску случайных пикселей
    mask = np.zeros((h, w), dtype=bool)
    indices = np.random.choice(h * w, num_pixels, replace=False)
    mask.flat[indices] = True
    
    # Применяем шум (делаем выбранные пиксели черными)
    noisy_marker[mask] = 0
    
    # Сохраняем зашумленный маркер
    cv2.imwrite('noisy_marker.png', noisy_marker)
    
    # Детектируем маркер
    try:
        original, contour_img, result_img, angle, area, distance = detect_marker('noisy_marker.png')
        
        # Выводим результаты
        print(f"ID маркера: 1")
        print(f"Угол поворота: {angle:.1f} градусов")
        print(f"Дистанция: {distance:.3f} метров")
        
        # Показываем зашумленный маркер
        cv2.imshow('noisy_marker', original)
        cv2.waitKey(1000)
        
        # Показываем контур
        cv2.imshow('contour_noisy_marker', contour_img)
        cv2.waitKey(1000)
        
        # Показываем прямоугольник
        cv2.imshow('detected_noisy_marker', result_img)
        cv2.waitKey(1000)
        
        # Закрываем все окна
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка при детектировании зашумленного маркера: {str(e)}")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    test_detector()
    test_size_variations()
    test_rotation()
    test_noise() 