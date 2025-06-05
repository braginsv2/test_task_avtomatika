import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random

def generate_aruco_marker(marker_id, size=200, border_size=20):
    """
    Генерирует базовый Aruco маркер по заданному ID
    
    Args:
        marker_id: ID маркера (0-65535)
        size: размер самого маркера в пикселях
        border_size: размер черной рамки маркера
    
    Returns:
        PIL.Image: Изображение маркера
    """
    # Проверяем, что ID в допустимом диапазоне для 16-битного числа
    if marker_id < 0 or marker_id > 65535:
        raise ValueError("ID маркера должен быть в диапазоне 0-65535")
    
    # Преобразуем ID в 16-битную двоичную строку
    binary_str = format(marker_id, '016b')
    
    # Создаем сетку 4x4
    grid = np.zeros((4, 4), dtype=int)
    
    # Заполняем сетку слева направо, сверху вниз
    for i in range(4):
        for j in range(4):
            bit_index = i * 4 + j
            grid[i][j] = int(binary_str[bit_index])
    
    # Создаем изображение маркера (черный квадрат)
    marker_total_size = size + 2 * border_size
    marker_img = Image.new('RGB', (marker_total_size, marker_total_size), color='black')
    draw = ImageDraw.Draw(marker_img)
    
    # Размер одной ячейки в сетке
    cell_size = size // 4
    
    # Рисуем внутренние ячейки маркера
    for i in range(4):
        for j in range(4):
            x1 = border_size + j * cell_size
            y1 = border_size + i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            # Если бит равен 1, рисуем белый квадрат, иначе оставляем черный
            color = 'white' if grid[i][j] == 1 else 'black'
            draw.rectangle([x1, y1, x2, y2], fill=color)
    
    return marker_img

def generate_quad_aruco_markers(marker_id, marker_size=200, border_size=20, 
                               spacing=50, background_padding=100):
    """
    Генерирует четыре одинаковых ArUco маркера, размещенных по квадрату
    
    Args:
        marker_id (int): ID маркера
        marker_size (int): Размер маркера в пикселях
        border_size (int): Размер черной рамки маркера
        spacing (int): Расстояние между маркерами в пикселях
        background_padding (int): Отступ от краев общего изображения
    
    Returns:
        PIL.Image: Изображение с четырьмя маркерами
    """
    # Генерируем один маркер
    single_marker = generate_aruco_marker(marker_id, marker_size, border_size)
    marker_total_size = marker_size + 2 * border_size
    
    # Вычисляем размеры итогового изображения
    # Два маркера + промежуток между ними + отступы с краев
    quad_width = 2 * marker_total_size + spacing + 2 * background_padding
    quad_height = 2 * marker_total_size + spacing + 2 * background_padding
    
    # Создаем белое изображение для размещения четырех маркеров
    quad_image = Image.new('RGB', (quad_width, quad_height), color='white')
    
    # Позиции для размещения маркеров (верхний левый, верхний правый, нижний левый, нижний правый)
    positions = [
        # Верхний левый
        (background_padding, background_padding),
        # Верхний правый
        (background_padding + marker_total_size + spacing, background_padding),
        # Нижний левый
        (background_padding, background_padding + marker_total_size + spacing),
        # Нижний правый
        (background_padding + marker_total_size + spacing, 
         background_padding + marker_total_size + spacing)
    ]
    
    # Размещаем маркеры в каждой позиции
    for i, (x, y) in enumerate(positions):
        quad_image.paste(single_marker, (x, y))
    
    return quad_image

def create_half_cropped_quad(quad_image):
    """
    Создает квад маркеров, обрезанный наполовину
    
    Args:
        quad_image: Исходное изображение квада
    
    Returns:
        PIL.Image: Обрезанный квад на белом фоне
    """
    width, height = quad_image.size
    
    # Создаем новое изображение того же размера с белым фоном
    cropped_quad = Image.new('RGB', (width, height), color='white')
    
    # Копируем только левую половину исходного квада
    left_half = quad_image.crop((0, 0, width // 2, height))
    cropped_quad.paste(left_half, (0, 0))
    
    return cropped_quad

def create_noisy_quad(quad_image, noise_percentage=30):
    """
    Создает квад маркеров с случайными белыми пикселями
    
    Args:
        quad_image: Исходное изображение квада
        noise_percentage: Процент пикселей для замены на белые
    
    Returns:
        PIL.Image: Квад с шумом
    """
    # Конвертируем в numpy array для удобства работы с пикселями
    img_array = np.array(quad_image)
    height, width, channels = img_array.shape
    
    # Вычисляем количество пикселей для замены
    total_pixels = height * width
    pixels_to_change = int(total_pixels * noise_percentage / 100)
    
    # Генерируем случайные координаты
    random.seed(42)  # Фиксируем seed для воспроизводимости
    positions = [(random.randint(0, height-1), random.randint(0, width-1)) 
                for _ in range(pixels_to_change)]
    
    # Создаем копию массива
    noisy_array = img_array.copy()
    
    # Заменяем случайные пиксели на белые
    for y, x in positions:
        noisy_array[y, x] = [255, 255, 255]  # Белый цвет
    
    # Конвертируем обратно в PIL Image
    return Image.fromarray(noisy_array)

def create_rotated_quad(quad_image, angle=30):
    """
    Создает квад маркеров, повернутый на заданный угол
    
    Args:
        quad_image: Исходное изображение квада
        angle: Угол поворота в градусах
    
    Returns:
        PIL.Image: Повернутый квад на белом фоне
    """
    # Поворачиваем изображение с белым фоном для заполнения пустых областей
    rotated_quad = quad_image.rotate(angle, fillcolor='white', expand=True)
    
    return rotated_quad

def generate_four_quad_variants(marker_id, marker_size=200, border_size=20, 
                               spacing=50, background_padding=100, noise_percentage=30, rotation_angle=30):
    """
    Генерирует четыре варианта квада ArUco маркеров:
    1. Нормальный квад (4 одинаковых маркера)
    2. Обрезанный наполовину квад
    3. Квад с шумом (30% белых пикселей)
    4. Повернутый квад (на 30 градусов)
    
    Args:
        marker_id (int): ID маркера
        marker_size (int): Размер маркера в пикселях
        border_size (int): Размер черной рамки маркера
        spacing (int): Расстояние между маркерами в пикселях
        background_padding (int): Отступ от краев общего изображения
        noise_percentage (int): Процент белых пикселей для третьего квада
        rotation_angle (int): Угол поворота для четвертого квада в градусах
    
    Returns:
        dict: Словарь с четырьмя вариантами квадов
    """
    # Генерируем базовый нормальный квад
    normal_quad = generate_quad_aruco_markers(
        marker_id=marker_id,
        marker_size=marker_size,
        border_size=border_size,
        spacing=spacing,
        background_padding=background_padding
    )
    
    # Создаем обрезанный квад
    cropped_quad = create_half_cropped_quad(normal_quad)
    
    # Создаем квад с шумом
    noisy_quad = create_noisy_quad(normal_quad, noise_percentage)
    
    # Создаем повернутый квад
    rotated_quad = create_rotated_quad(normal_quad, rotation_angle)
    
    marker_total_size = marker_size + 2 * border_size
    quad_width = 2 * marker_total_size + spacing + 2 * background_padding
    quad_height = 2 * marker_total_size + spacing + 2 * background_padding
    
    print(f"Созданы четыре варианта квада маркеров ID: {marker_id}")
    print(f"1. Нормальный квад: 4 одинаковых маркера")
    print(f"2. Обрезанный квад: левая половина нормального квада")
    print(f"3. Квад с шумом: {noise_percentage}% пикселей заменены на белые")
    print(f"4. Повернутый квад: поворот на {rotation_angle} градусов")
    print(f"Размер базового квада: {quad_width}x{quad_height}")
    
    return {
        'normal': normal_quad,
        'cropped': cropped_quad,
        'noisy': noisy_quad,
        'rotated': rotated_quad
    }

def save_four_quad_variants(marker_id, marker_size=200, border_size=20, 
                           spacing=50, background_padding=100, noise_percentage=30, rotation_angle=30):
    """
    Генерирует и сохраняет четыре варианта квада ArUco маркеров
    
    Args:
        marker_id (int): ID маркера
        marker_size (int): Размер маркера в пикселях
        border_size (int): Размер черной рамки маркера
        spacing (int): Расстояние между маркерами в пикселях
        background_padding (int): Отступ от краев общего изображения
        noise_percentage (int): Процент белых пикселей для шумного квада
        rotation_angle (int): Угол поворота для четвертого квада в градусах
    
    Returns:
        dict: Словарь с информацией о сохраненных файлах
    """
    # Генерируем четыре варианта
    quads = generate_four_quad_variants(
        marker_id=marker_id,
        marker_size=marker_size,
        border_size=border_size,
        spacing=spacing,
        background_padding=background_padding,
        noise_percentage=noise_percentage,
        rotation_angle=rotation_angle
    )
    
    # Определяем имена файлов
    filenames = {
        'normal': f"quad_aruco_{marker_id}_normal.png",
        'cropped': f"quad_aruco_{marker_id}_cropped.png",
        'noisy': f"quad_aruco_{marker_id}_noisy_{noise_percentage}pct.png",
        'rotated': f"quad_aruco_{marker_id}_rotated_{rotation_angle}deg.png"
    }
    
    # Сохраняем каждый вариант
    saved_files = {}
    for variant_name, quad_image in quads.items():
        filename = filenames[variant_name]
        quad_image.save(filename)
        print(f"Сохранен {variant_name} квад как: {filename}")
        
        saved_files[variant_name] = {
            'image': quad_image,
            'filename': filename
        }
    
    return saved_files

def display_four_quad_variants(marker_id, marker_size=200, border_size=20, 
                              spacing=50, background_padding=100, noise_percentage=30, rotation_angle=30):
    """
    Генерирует и отображает четыре варианта квада ArUco маркеров
    
    Args:
        marker_id (int): ID маркера
        marker_size (int): Размер маркера в пикселях
        border_size (int): Размер черной рамки маркера
        spacing (int): Расстояние между маркерами в пикселях
        background_padding (int): Отступ от краев общего изображения
        noise_percentage (int): Процент белых пикселей для шумного квада
        rotation_angle (int): Угол поворота для четвертого квада в градусах
    
    Returns:
        dict: Словарь с четырьмя вариантами квадов
    """
    quads = generate_four_quad_variants(
        marker_id=marker_id,
        marker_size=marker_size,
        border_size=border_size,
        spacing=spacing,
        background_padding=background_padding,
        noise_percentage=noise_percentage,
        rotation_angle=rotation_angle
    )
    
    # Отображаем все четыре варианта
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    titles = [
        f'1. Нормальный квад\n(ID: {marker_id})',
        f'2. Обрезанный квад\n(левая половина)',
        f'3. Квад с шумом\n({noise_percentage}% белых пикселей)',
        f'4. Повернутый квад\n(поворот на {rotation_angle}°)'
    ]
    
    variants = ['normal', 'cropped', 'noisy', 'rotated']
    
    for i, (variant, title) in enumerate(zip(variants, titles)):
        row = i // 2
        col = i % 2
        axes[row, col].imshow(quads[variant])
        axes[row, col].set_title(title, fontsize=12, pad=10)
        axes[row, col].axis('off')
    
    plt.suptitle(f'Четыре варианта квада ArUco маркеров (ID: {marker_id})', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return quads

def generate_multiple_four_quad_variants(marker_ids, marker_size=200, border_size=20, 
                                        spacing=50, background_padding=100, noise_percentage=30, rotation_angle=30):
    """
    Генерирует четыре варианта квадов для нескольких ID маркеров
    
    Args:
        marker_ids (list): Список ID маркеров
        marker_size (int): Размер маркера в пикселях
        border_size (int): Размер черной рамки маркера
        spacing (int): Расстояние между маркерами в пикселях
        background_padding (int): Отступ от краев общего изображения
        noise_percentage (int): Процент белых пикселей для шумного квада
        rotation_angle (int): Угол поворота для четвертого квада в градусах
    
    Returns:
        dict: Словарь с информацией о всех сгенерированных квадах
    """
    all_quads = {}
    
    for marker_id in marker_ids:
        print(f"\n{'='*70}")
        print(f"Генерируем четыре варианта квадов для ID: {marker_id}")
        print(f"{'='*70}")
        
        saved_files = save_four_quad_variants(
            marker_id=marker_id,
            marker_size=marker_size,
            border_size=border_size,
            spacing=spacing,
            background_padding=background_padding,
            noise_percentage=noise_percentage,
            rotation_angle=rotation_angle
        )
        
        all_quads[marker_id] = saved_files
    
    print(f"\n{'='*70}")
    print(f"Создано {len(all_quads)} наборов квадов (по 4 варианта каждый)")
    print(f"Всего файлов: {len(all_quads) * 4}")
    print(f"{'='*70}")
    
    return all_quads

def create_comparison_grid(marker_ids, marker_size=200, border_size=20, 
                         spacing=50, background_padding=100, noise_percentage=30, rotation_angle=30):
    """
    Создает сравнительную сетку всех вариантов квадов
    
    Args:
        marker_ids (list): Список ID маркеров
        marker_size (int): Размер маркера в пикселях
        border_size (int): Размер черной рамки маркера
        spacing (int): Расстояние между маркерами в пикселях
        background_padding (int): Отступ от краев общего изображения
        noise_percentage (int): Процент белых пикселей для шумного квада
        rotation_angle (int): Угол поворота для четвертого квада в градусах
    """
    num_ids = len(marker_ids)
    
    # Создаем сетку: строки = количество ID, колонки = 4 варианта
    fig, axes = plt.subplots(num_ids, 4, figsize=(16, 4 * num_ids))
    
    # Если только один ID, axes нужно преобразовать
    if num_ids == 1:
        axes = axes.reshape(1, -1)
    
    variant_names = ['normal', 'cropped', 'noisy', 'rotated']
    variant_titles = ['Нормальный', 'Обрезанный', f'Шум {noise_percentage}%', f'Поворот {rotation_angle}°']
    
    for row, marker_id in enumerate(marker_ids):
        quads = generate_four_quad_variants(
            marker_id=marker_id,
            marker_size=marker_size,
            border_size=border_size,
            spacing=spacing,
            background_padding=background_padding,
            noise_percentage=noise_percentage,
            rotation_angle=rotation_angle
        )
        
        for col, (variant, title) in enumerate(zip(variant_names, variant_titles)):
            ax = axes[row, col] if num_ids > 1 else axes[col]
            ax.imshow(quads[variant])
            ax.set_title(f'{title}\n(ID: {marker_id})', fontsize=10)
            ax.axis('off')
    
    plt.suptitle('Сравнение всех вариантов квадов ArUco маркеров', fontsize=16)
    plt.tight_layout()
    plt.show()

# Пример использования
if __name__ == "__main__":
    print("=== Генератор четырех вариантов квадов ArUco маркеров ===")
    print("Создает для каждого ID четыре квада:")
    print("1. Нормальный квад (4 одинаковых маркера)")
    print("2. Обрезанный квад (левая половина)")
    print("3. Квад с шумом (30% случайных белых пикселей)")
    print("4. Повернутый квад (поворот на 30 градусов)")
    
    # Параметры генерации
    marker_id = 12
    marker_size = 200
    border_size = 15
    spacing = 15
    background_padding = 50
    noise_percentage = 30
    rotation_angle = -40
    
    print(f"\nГенерируем четыре варианта квадов для ID: {marker_id}")
    print(f"Размер маркера: {marker_size}px")
    print(f"Размер рамки: {border_size}px")
    print(f"Промежуток между маркерами: {spacing}px")
    print(f"Отступ от краев: {background_padding}px")
    print(f"Процент шума: {noise_percentage}%")
    print(f"Угол поворота: {rotation_angle}°")
    
    # Генерируем и сохраняем четыре варианта
    saved_files = save_four_quad_variants(
        marker_id=marker_id,
        marker_size=marker_size,
        border_size=border_size,
        spacing=spacing,
        background_padding=background_padding,
        noise_percentage=noise_percentage,
        rotation_angle=rotation_angle
    )
    
    # Отображаем результат
    print("\nОтображаем результат...")
    display_four_quad_variants(
        marker_id=marker_id,
        marker_size=marker_size,
        border_size=border_size,
        spacing=spacing,
        background_padding=background_padding,
        noise_percentage=noise_percentage,
        rotation_angle=rotation_angle
    )
    
    # Пример генерации для нескольких ID
    print(f"\n{'='*70}")
    print("Пример: генерация четырех вариантов для нескольких ID")
    
   
    
    
    
    print("Все варианты квадов маркеров сохранены в текущей директории")
    print("Формат файлов:")
    print("- *_normal.png - нормальные квады")
    print("- *_cropped.png - обрезанные квады")
    print("- *_noisy_30pct.png - квады с шумом")
    print("- *_rotated_30deg.png - повернутые квады")
