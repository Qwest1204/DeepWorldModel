import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

def create_vae_animation(npz_path, vae_model, save_path="vae_reconstruction.gif", num_frames=100):
    """
    npz_path: путь к файлу с данными (предполагаем ключ 'obs')
    vae_model: ваша загруженная модель (torch.nn.Module)
    save_path: куда сохранить гифку
    num_frames: сколько кадров анимировать
    """

    # 1. Загрузка данных
    print(f"Загрузка данных из {npz_path}...")
    data = np.load(npz_path)

    # Предполагаем, что картинки лежат по ключу 'obs' или 'observations'
    # Если у вас другой ключ, поменяйте его здесь
    key = 'obs' if 'obs' in data else list(data.keys())[0]
    observations = data[key] # Ожидаем форму (N, 96, 96, 3) или (N, 64, 64, 3)

    # 2. Подготовка модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model.to(device)
    vae_model.eval() # Переключаем в режим оценки (выключает dropout и т.д.)

    # 3. Настройка графика
    fig, ax = plt.subplots()
    ax.axis('off') # Убираем оси координат

    # Инициализация пустой картинки
    # Берем первый кадр просто чтобы узнать размер
    img_plot = ax.imshow(observations[0])

    print("Генерация анимации...")

    def update(frame_idx):
        # --- ПРЕПРОЦЕССИНГ ---
        # Берем кадр из датасета. Он сейчас (H, W, C) и 0-255 uint8
        raw_frame = observations[frame_idx]

        # Конвертируем в формат PyTorch: (1, C, H, W) и нормализуем 0-1
        # permute(2, 0, 1) делает из HWC -> CHW
        input_tensor = torch.from_numpy(raw_frame).permute(2, 0, 1).float() / 255.0

        # Добавляем Batch dimension -> (1, C, H, W)
        x = input_tensor.unsqueeze(0).to(device)

        # --- ВАШ КОД + ИНФЕРЕНС ---
        with torch.no_grad(): # Отключаем градиенты для скорости
            # ВАШ СНИППЕТ ВСТАВЛЕН СЮДА
            # x подается в VAE, на выходе реконструированная картинка

            # Примечание: Ваш сниппет возвращает PIL Image.
            # Matplotlib хочет numpy array. Поэтому оборачиваем в np.array()
            reconstructed_pil = Image.fromarray(
                (vae_model(x)[0].squeeze(0).cpu().detach().numpy().transpose(1,2,0).astype(np.float32)*255).astype(np.uint8)
            )

            reconstructed_img = np.array(reconstructed_pil)

        # Обновляем картинку на графике
        img_plot.set_data(reconstructed_img)
        ax.set_title(f"Frame: {frame_idx}")
        return [img_plot]

    # Создание анимации
    # frames=num_frames ограничивает длину, чтобы не рендерить весь датасет часами
    ani = FuncAnimation(fig, update, frames=range(num_frames), blit=True)

    # Сохранение
    print(f"Сохранение в {save_path}...")
    ani.save(save_path, fps=30, writer='pillow') # 'pillow' сохраняет gif
    print("Готово!")
    plt.close()