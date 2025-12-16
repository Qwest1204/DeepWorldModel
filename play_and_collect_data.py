import gymnasium as gym
import pygame
import numpy as np

def play_car_racing(save=True):
    # Создаем среду с режимом рендеринга для человека
    env = gym.make("CarRacing-v3", render_mode="human")

    # Сброс среды для начала
    observation, info = env.reset()

    running = True
    print("Управление:")
    print("СТРЕЛКИ ВЛЕВО/ВПРАВО - Поворот")
    print("СТРЕЛКА ВВЕРХ - Газ")
    print("СТРЕЛКА ВНИЗ - Тормоз")
    print("ESC - Выход")

    observations = []
    actions = []
    i = 0

    while running:
        # 1. Обработка событий окна (чтобы оно не зависло и можно было выйти)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # 2. Чтение нажатых клавиш
        keys = pygame.key.get_pressed()

        # 3. Формирование действия
        # CarRacing принимает массив: [steering, gas, brake]
        # steering: от -1 (влево) до 1 (вправо)
        # gas: от 0 до 1
        # brake: от 0 до 1

        steering = 0.0
        gas = 0.0
        brake = 0.0

        if keys[pygame.K_LEFT]:
            steering = -1.0
        if keys[pygame.K_RIGHT]:
            steering = 1.0
        if keys[pygame.K_UP]:
            gas = 1.0
        if keys[pygame.K_DOWN]:
            brake = 0.8 # Немного мягче тормоз

        action = np.array([steering, gas, brake], dtype=np.float32)

        # 4. Шаг в среде
        # next_obs - картинка (пиксели)
        # reward - награда
        # terminated - игра окончена (вылетел или проехал)
        # truncated - время вышло
        next_obs, reward, terminated, truncated, info = env.step(action)
        observations.append(next_obs)
        actions.append(action)
        if terminated or truncated:
            print("Эпизод закончен! Сохранение наблюдений Перезапуск...")
            if save == True:
                np.savez_compressed(f"data/car_racing_data_ep_{i}.npz", obs=observations, act=actions)
            i = i + 1

            env.reset()

    env.close()
    pygame.quit()

if __name__ == "__main__":
    play_car_racing(False)