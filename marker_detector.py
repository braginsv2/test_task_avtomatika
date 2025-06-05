import cv2
import numpy as np
import matplotlib.pyplot as plt

class ArucoDetector:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
    
    def preprocess_image(self, image):
        """
        Предобработка изображения для детекции
        """
        # Конвертируем в grayscale если цветное
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Применяем адаптивную бинаризацию
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary, gray

    def find_and_normalize_marker(self, image, target_size=240, debug=False):
        """
        Находит маркер в изображении и нормализует его размер
        """
        binary, gray = self.preprocess_image(image)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours, key=cv2.contourArea, reverse=True)
        
        
        
        rect = cv2.minAreaRect(contours[1])
        (center, (width, height), angle) = rect
       
        
        if abs(angle) > 1:  # Поворачиваем только если угол значительный
                # Поворачиваем в обратную сторону для компенсации
                if (abs(angle))>45:
                    corrected_angle = 90-angle
                    rotation_angle = -corrected_angle
                    image = rotate_image(image, rotation_angle)
                    self.rotation=corrected_angle
                if (abs(angle))<45:
                    corrected_angle = angle
                    rotation_angle =corrected_angle
                    image = rotate_image(image, rotation_angle)
                    self.rotation=corrected_angle        
        else:
            self.rotation=angle
        
        
        binary, gray = self.preprocess_image(image)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[1])
        image = image[y:y+h, x:x+w]
        self.big_contour = contours[1] 
        
        # Предобработка для поиска контуров
        binary, gray = self.preprocess_image(image)
        
        # Находим все контуры
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug:
            print(f"Поиск маркера в изображении {image.shape}")
            print(f"Найдено контуров: {len(contours)}")
        
        # Ищем внешние контуры
        external_contours = []
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] == -1:  # внешний контур
                    external_contours.append((i, contours[i]))
        else:
            external_contours = [(i, contours[i]) for i in range(len(contours))]
        
        # Находим наибольший прямоугольный контур
        best_contour = None
        best_area = 0
        image_area = image.shape[0] * image.shape[1]
        
        for idx, (i, contour) in enumerate(external_contours):
            area = cv2.contourArea(contour)
            
            # Фильтр по минимальной площади
            if area < image_area * 0.01:  # минимум 1% от изображения
                continue
            
            # Аппроксимация контура
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Должно быть примерно 4 вершины
            if len(approx) < 4 or len(approx) > 8:
                continue
            
            # Проверяем соотношение сторон
            rect = cv2.minAreaRect(contour)
            (center, (width, height), angle) = rect
            
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 3.0:  # очень гибкое ограничение
                    continue
            
            if area > best_area:
                best_area = area
                best_contour = approx
                
                if debug:
                    print(f"Новый лучший контур: площадь {area}, вершин {len(approx)}")
        
        if best_contour is None:
            if debug:
                print("Маркер не найден")
            return None
        
        if debug:
            print(f"Найден маркер с площадью {best_area}")
        
        # Получаем ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Добавляем небольшой отступ
        margin = max(w, h) * 0.1
        x = max(0, int(x - margin))
        y = max(0, int(y - margin))
        w = min(image.shape[1] - x, int(w + 2 * margin))
        h = min(image.shape[0] - y, int(h + 2 * margin))
        
        # Извлекаем область маркера
        marker_region = image[y:y+h, x:x+w]
        
        if debug:
            print(f"Область маркера: {marker_region.shape}")
            #cv2.imwrite('debug_marker_region.png', marker_region)
        
        # Для очень маленьких изображений увеличиваем target_size
        min_dimension = min(image.shape[0], image.shape[1])
        if min_dimension < 150:
            target_size = max(target_size, 320)  # Увеличиваем для лучшего качества
            if debug:
                print(f"Маленькое изображение ({min_dimension}px), увеличиваем target_size до {target_size}")
        
        # Изменяем размер к целевому (квадрат)
        normalized_marker = cv2.resize(marker_region, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        
        if debug:
            print(f"Нормализованный маркер: {normalized_marker.shape}")
            #cv2.imwrite('debug_normalized_marker.png', normalized_marker)
        
        
        return normalized_marker

    def find_contours_normalized(self, binary_image, show_debug=False):
        """
        Находит контуры в нормализованном изображении (старая проверенная логика)
        """
        # Находим контуры
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if show_debug:
            print(f"Найдено всех контуров: {len(contours)}")
        
        # Если иерархия есть, ищем внешние контуры
        external_contours = []
        if hierarchy is not None:
            # hierarchy[i] = [next, previous, first_child, parent]
            # Внешние контуры имеют parent = -1
            for i, h in enumerate(hierarchy[0]):
                if h[3] == -1:  # parent == -1 означает внешний контур
                    external_contours.append((i, contours[i]))
        else:
            # Если иерархии нет, берем все контуры
            external_contours = [(i, contours[i]) for i in range(len(contours))]
        
        if show_debug:
            print(f"Найдено внешних контуров: {len(external_contours)}")
        
        candidates = []
        
        for idx, (i, contour) in enumerate(external_contours):
            area = cv2.contourArea(contour)
            

            if show_debug:
                print(f"Внешний контур {idx}: площадь = {area}")
            
            # Фильтр по площади - проверенные значения для нормализованных изображений
            if area < 1000 or area > 105000:
                if show_debug:
                    print(f"  Контур {idx} отклонен по площади")
                continue
            
            # Аппроксимация контура
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if show_debug:
                print(f"  Вершин после аппроксимации: {len(approx)}")
            
            # Должно быть примерно 4 вершины (квадрат)
            if len(approx) != 4:
                if show_debug:
                    print(f"  Контур {idx} отклонен: не 4 вершины")
                continue
            
            # Проверяем соотношение сторон (должно быть близко к квадрату)
            rect = cv2.minAreaRect(contour)
            (center, (width, height), angle) = rect
            
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                if show_debug:
                    print(f"  Соотношение сторон: {aspect_ratio:.3f}")
                
                if aspect_ratio > 1.5:
                    if show_debug:
                        print(f"  Контур {idx} отклонен: плохое соотношение сторон")
                    continue
            
            # Проверяем наличие внутренних контуров (должна быть внутренняя структура)
            inner_contours = 0
            if hierarchy is not None:
                for j, h in enumerate(hierarchy[0]):
                    if h[3] == i:  # parent == i означает дочерний контур
                        inner_contours += 1
            
            if show_debug:
                print(f"  Внутренних контуров: {inner_contours}")
            
            if inner_contours == 0:
                if show_debug:
                    print(f"  Контур {idx} отклонен: нет внутренних контуров")
                continue
            
            if show_debug:
                print(f"  Контур {idx} добавлен в кандидаты")
            
            candidates.append(approx)
        
        return candidates

    def order_points(self, pts):
        """
        Упорядочивает точки четырехугольника: top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Сумма координат: top-left имеет наименьшую сумму, bottom-right - наибольшую
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        # Разность координат: top-right имеет наименьшую разность (x-y), bottom-left - наибольшую
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect

    def perspective_transform(self, image, pts, size=200):
        """
        Применяет перспективное преобразование для выравнивания маркера
        """
        rect = self.order_points(pts.reshape(4, 2))
        
        # Определяем целевые точки для квадрата
        dst = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ], dtype="float32")
        
        # Вычисляем матрицу трансформации
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (size, size))
        
        return warped

    def extract_grid(self, warped_image, debug=False):
        """
        Извлекает сетку 4x4 из выровненного изображения маркера
        """
        if debug:
            print(f"Размер выровненного изображения: {warped_image.shape}")
            print(f"Диапазон значений: {warped_image.min()} - {warped_image.max()}")
            
            # Сохраняем выровненное изображение для анализа
            #cv2.imwrite('debug_warped_original.png', warped_image)
        
        # Убираем рамку (примерно 10% с каждой стороны - более консервативно)
        h, w = warped_image.shape
        border = int(min(h, w) * 0.1)
        inner = warped_image[border:h-border, border:w-border]
        
        if debug:
            print(f"Размер внутренней области после удаления рамки: {inner.shape}")
            print(f"Диапазон значений внутренней области: {inner.min()} - {inner.max()}")
            #cv2.imwrite('debug_inner_area.png', inner)
        
        # Делим на сетку 4x4
        cell_h = inner.shape[0] // self.grid_size
        cell_w = inner.shape[1] // self.grid_size
        
        if debug:
            print(f"Размер ячейки: {cell_h} x {cell_w}")
        
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Собираем все средние значения для автоматического определения порога
        all_means = []
        cell_data = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Извлекаем ячейку - берем центральные 80% от каждой ячейки
                margin_h = cell_h // 10
                margin_w = cell_w // 10
                
                y1 = i * cell_h + margin_h
                y2 = (i + 1) * cell_h - margin_h
                x1 = j * cell_w + margin_w
                x2 = (j + 1) * cell_w - margin_w
                
                # Проверяем границы
                y1 = max(0, y1)
                x1 = max(0, x1)
                y2 = min(inner.shape[0], y2)
                x2 = min(inner.shape[1], x2)
                
                cell = inner[y1:y2, x1:x2]
                
                if cell.size == 0:
                    mean_val = 0.0
                else:
                    mean_val = np.mean(cell)
                
                all_means.append(mean_val)
                cell_data.append((i, j, x1, x2, y1, y2, mean_val, cell))
                
                if debug:
                    print(f"Ячейка [{i}][{j}]: координаты ({x1}:{x2}, {y1}:{y2}), среднее значение: {mean_val:.1f}")
                    # Сохраняем каждую ячейку для анализа
                    
        
        # Автоматическое определение порога
        all_means = np.array(all_means)
        if debug:
            print(f"Все средние значения: {all_means}")
            print(f"Мин: {all_means.min():.1f}, Макс: {all_means.max():.1f}, Медиана: {np.median(all_means):.1f}")
        
        # Улучшенный алгоритм определения порога
        if all_means.max() - all_means.min() > 50:  # Если есть достаточный контраст
            # Используем среднее между минимумом и максимумом
            threshold = (all_means.min() + all_means.max()) / 2
            
            # Альтернативно: используем метод Оцу для автоматического определения
            # Создаем гистограмму значений
            non_zero_means = all_means[all_means > 0]
            if len(non_zero_means) >= 2:
                # Если есть и черные (0) и не-черные значения
                unique_vals = np.unique(all_means)
                if len(unique_vals) >= 3:  # есть 0, низкие и высокие значения
                    # Находим "разрыв" в значениях
                    sorted_vals = np.sort(unique_vals)
                    max_gap = 0
                    best_threshold = threshold
                    
                    for i in range(len(sorted_vals) - 1):
                        gap = sorted_vals[i+1] - sorted_vals[i]
                        if gap > max_gap:
                            max_gap = gap
                            best_threshold = (sorted_vals[i] + sorted_vals[i+1]) / 2
                    
                    if max_gap > 50:  # Если есть значительный разрыв
                        threshold = best_threshold
                        
                # Дополнительная проверка для случаев с плохим качеством
                # Если у нас есть промежуточные значения (не 0 и не 255), 
                # может быть стоит использовать более низкий порог
                intermediate_values = all_means[(all_means > 10) & (all_means < 200)]
                if len(intermediate_values) > 0 and len(non_zero_means) >= len(intermediate_values):
                    # Есть промежуточные значения - возможно низкое качество
                    # Используем 75-й процентиль от всех ненулевых значений
                    if len(non_zero_means) >= 2:
                        threshold = min(threshold, np.percentile(non_zero_means, 87))
                        
        else:
            threshold = 50  # Фиксированный низкий порог для слабоконтрастных изображений
            
        if debug:
            print(f"Выбранный порог: {threshold:.1f}")
        
        # Применяем порог
        for idx, (i, j, x1, x2, y1, y2, mean_val, cell) in enumerate(cell_data):
            grid[i][j] = 1 if mean_val > threshold else 0
            
            if debug:
                print(f"Ячейка [{i}][{j}]: {mean_val:.1f} > {threshold:.1f} = {'БЕЛАЯ' if grid[i][j] == 1 else 'ЧЕРНАЯ'}")
        
        if debug:
            print(f"Полученная сетка:\n{grid}")
        
        return grid

    def is_valid_marker(self, grid, warped_image, debug=False):
        """
        Проверяет, является ли извлеченная сетка валидным маркером
        """
        if debug:
            print("Проверка валидности маркера...")
            print(f"Сетка:\n{grid}")
        
        # Проверяем количество уникальных значений
        unique_values = len(np.unique(grid))
        if debug:
            print(f"Уникальных значений в сетке: {unique_values}")
        
        if unique_values < 2:
            if debug:
                print("Отклонено: все ячейки одинаковые")
            return False
        
        # Проверяем соотношение белых и черных ячеек
        white_cells = np.sum(grid == 1)
        black_cells = np.sum(grid == 0)
        
        if debug:
            print(f"Белых ячеек: {white_cells}, черных ячеек: {black_cells}")
        
        # Должно быть достаточно разнообразия
        total_cells = self.grid_size * self.grid_size
        if white_cells < 1 or white_cells > total_cells - 1:
            if debug:
                print("Отклонено: неподходящее соотношение белых/черных ячеек")
            return False
        
        if debug:
            print("Маркер прошел проверку валидности")
        
        return True

    def grid_to_id(self, grid):
        """
        Преобразует сетку 4x4 в ID маркера
        """
        binary_str = ""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                binary_str += str(grid[i][j])
        
        # Преобразуем двоичную строку в число
        marker_id = int(binary_str, 2)
        return marker_id, binary_str

    def try_all_orientations(self, grid, debug=False):
        """
        Пробует все 4 ориентации маркера и возвращает результат с максимальным ID
        """
        orientations = []
        
        # Исходная ориентация
        orientations.append(grid.copy())
        
        # Поворот на 90 градусов (3 раза)
        current = grid.copy()
        for _ in range(3):
            current = np.rot90(current)
            orientations.append(current.copy())
        
        if debug:
            print("Проверяем все ориентации:")
        
        results = []
        for i, orientation in enumerate(orientations):
            marker_id, binary_str = self.grid_to_id(orientation)
            results.append((marker_id, binary_str, orientation))
            
            if debug:
                print(f"Ориентация {i} (поворот на {i*90}°):")
                print(f"  Сетка:\n{orientation}")
                print(f"  ID: {marker_id}")
                print(f"  Двоичный код: {binary_str}")
        
        # Возвращаем ориентацию с максимальным разнообразием или специфичным паттерном
        # Для отладки пока возвращаем все варианты
        return results

    def detect_markers(self, image, show_debug=False):
        """
        Основная функция детекции маркеров с нормализацией размера
        """
        # Сначала находим и нормализуем маркер
        normalized_marker = self.find_and_normalize_marker(image, target_size=240, debug=show_debug)
        
        if normalized_marker is None:
            if show_debug:
                print("Маркер не найден в изображении")
            return [] if not show_debug else ([], image)
        
        if show_debug:
            print("Маркер найден и нормализован, продолжаем детекцию...")
        
        # Теперь применяем проверенный алгоритм к нормализованному изображению
        # Предобработка
        binary, gray = self.preprocess_image(normalized_marker)
        
        # Поиск контуров с проверенными ограничениями
        candidates = self.find_contours_normalized(binary, show_debug)
        
        detected_markers = []
        
        if show_debug:
            debug_image = normalized_marker.copy()
            if len(debug_image.shape) == 2:
                debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
            print(f"Найдено кандидатов в нормализованном изображении: {len(candidates)}")
        
        for i, candidate in enumerate(candidates):
            try:
                if show_debug:
                    print(f"\nОбработка кандидата {i}:")
                
                # Выравнивание перспективы - используем исходное grayscale изображение
                warped = self.perspective_transform(gray, candidate)
                
                if show_debug:
                    print(f"  Перспективное преобразование выполнено")
                
                # Извлечение сетки
                grid = self.extract_grid(warped, debug=show_debug)
                
                if show_debug:
                    print(f"  Извлеченная сетка:\n{grid}")
                
                # Проверяем, является ли это действительно маркером
                if not self.is_valid_marker(grid, warped, debug=show_debug):
                    if show_debug:
                        print(f"  Кандидат {i} не прошел проверку валидности")
                    continue
                
                # Пробуем все ориентации
                orientation_results = self.try_all_orientations(grid, debug=show_debug)
                
                # Берем исходную ориентацию (проверенный подход)
                marker_id, binary_str, final_grid = orientation_results[0]
                
                if show_debug:
                    print(f"  Выбрана исходная ориентация с ID: {marker_id}")
                    print(f"  Финальная сетка:\n{final_grid}")
                    print(f"  Двоичный код: {binary_str}")
                
                # Вычисляем центр маркера (в нормализованных координатах)
                M = cv2.moments(candidate)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 120, 120  # центр нормализованного изображения 240x240
                
                detected_markers.append({
                    'id': marker_id,
                    'binary': binary_str,
                    'grid': final_grid,
                    'contour': candidate,
                    'center': (cx, cy),
                    'warped': warped
                })
                
                if show_debug:
                    # Рисуем контур
                    cv2.drawContours(debug_image, [candidate], -1, (0, 255, 0), 2)
                    # Подписываем ID
                    cv2.putText(debug_image, f'ID: {marker_id}', 
                              (cx-30, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (255, 0, 0), 2)
                    # Отмечаем центр
                    cv2.circle(debug_image, (cx, cy), 3, (0, 0, 255), -1)
                
            except Exception as e:
                if show_debug:
                    print(f"Ошибка обработки кандидата {i}: {e}")
                continue
        
        if show_debug:
            return detected_markers, debug_image, self.rotation
        
        return detected_markers, self.rotation

def rotate_image( image, angle):
        """
        Поворачивает изображение на заданный угол
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Создаем матрицу поворота
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Поворачиваем изображение
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated

def detect_from_camera():
    """
    Детектирует маркеры с камеры в реальном времени
    """
    cap = cv2.VideoCapture(0)
    detector = ArucoDetector()
    
    print("Нажмите 'q' для выхода")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детектируем маркеры
        markers = detector.detect_markers(frame)
        
    cap.release()
    cv2.destroyAllWindows()

def detect_from_file(image_path, show_debug=True):
    """
    Детектирует маркеры из файла изображения
    """
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None
    
    print(f"Загружено изображение размером: {image.shape}")
    
    # Создаем детектор
    detector = ArucoDetector()
    
    # Показываем этапы предобработки только если включен debug
    if show_debug:
        # Сначала показываем поиск маркера в исходном изображении
        normalized_marker = detector.find_and_normalize_marker(image, target_size=240, debug=True)
        
        if normalized_marker is None:
            print("Маркер не найден в изображении")
            return []
        
        # Теперь показываем работу с нормализованным изображением
        binary, gray = detector.preprocess_image(normalized_marker)
    
        # Показываем отфильтрованные контуры
        candidates = detector.find_contours_normalized(binary, show_debug=True)
        candidate_image = cv2.cvtColor(normalized_marker, cv2.COLOR_BGR2RGB)
        for i, candidate in enumerate(candidates):
            cv2.drawContours(candidate_image, [candidate], -1, (255, 0, 0), 3)
            # Добавляем номер кандидата
            M = cv2.moments(candidate)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(candidate_image, str(i), (cx-10, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Если есть кандидаты, показываем первый выровненный
        if candidates:
            try:
                warped = detector.perspective_transform(gray, candidates[0])
                
                # Показываем извлеченную сетку
                grid = detector.extract_grid(warped, debug=True)
                print(f"\nИзвлеченная сетка из первого кандидата:")
                print(grid)
                
            except Exception as e:
                print(f"Ошибка при обработке первого кандидата: {e}")
        
        plt.tight_layout()
        #plt.show()
    
    # Детектируем маркеры с новым двухэтапным подходом
    if show_debug:
        markers, debug_image, corrected_angle = detector.detect_markers(image, show_debug=True)
    else:
        markers = detector.detect_markers(image)
    
    # Выводим информацию о найденных маркерах
    print(f"\nИтоговый результат: найдено маркеров: {len(markers)}")
    for i, marker in enumerate(markers):
        print(f"\nМаркер {i + 1}:")
        print(f"ID: {marker['id']}")
        print(f"angle: {corrected_angle}")
        print(f"Двоичный код: {marker['binary']}")
        print(f"Центр: {marker['center']}")
        print(f"Сетка:\n{marker['grid']}")
    
    return markers

# Пример использования
if __name__ == "__main__":   
    # Пример детекции из файла
    markers = detect_from_file('quad_aruco_673_normal.png', show_debug=True)
