import numpy as np
import cv2
import argparse
from PIL import Image, ImageDraw, ImageFont

class MarkerGenerator:
    def __init__(self, size: int = 400, border_size: int = 50):
        self.size = size
        self.border_size = border_size
        
    def generate_marker(self, marker_id: int) -> np.ndarray:
        """Генерация маркера с заданным ID"""
        # Создаем изображение
        image = Image.new('L', (self.size, self.size), 255)
        draw = ImageDraw.Draw(image)
        
        # Рисуем внешнюю рамку
        draw.rectangle([0, 0, self.size-1, self.size-1], outline=0, width=self.border_size)
        
        # Рисуем угловые маркеры
        corner_size = self.border_size * 3
        for i in range(4):
            x = (i % 2) * (self.size - corner_size)
            y = (i // 2) * (self.size - corner_size)
            draw.rectangle([x, y, x + corner_size - 1, y + corner_size - 1], 
                         fill=0)
        
        # Рисуем ID маркера в центре
        # Преобразуем ID в бинарный код
        binary_id = format(marker_id, 'b').zfill(4)
        
        # Размер одной ячейки для бита
        cell_size = (self.size - 4 * self.border_size) // 2
        
        # Рисуем биты
        for i, bit in enumerate(binary_id):
            if bit == '1':
                x = (i % 2) * cell_size + 2 * self.border_size
                y = (i // 2) * cell_size + 2 * self.border_size
                draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], 
                             fill=0)
        
        # Конвертируем PIL Image в numpy array
        return np.array(image)
    
    def save_marker(self, marker_id: int, filename: str):
        """Сохранение маркера в файл"""
        marker = self.generate_marker(marker_id)
        Image.fromarray(marker).save(filename)

def main():
    parser = argparse.ArgumentParser(description='Генератор визуальных меток')
    parser.add_argument('--id', type=int, required=True, help='ID метки')
    parser.add_argument('--output', type=str, required=True, help='Путь для сохранения метки')
    
    args = parser.parse_args()
    
    generator = MarkerGenerator()
    generator.save_marker(args.id, args.output)
    print(f'Метка сохранена в {args.output}')

if __name__ == '__main__':
    main() 