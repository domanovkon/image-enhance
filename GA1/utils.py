import matplotlib.pyplot as plt
from improvement_criteria import *

# -----------------------------------------------------------
# Строит график зависимости поколения от значения
# фитнес-функции
# -----------------------------------------------------------
def plot_generations_graph(fitness_values_array):
    plt.figure(figsize=(10, 10))
    plt.plot([i for i in range(len(fitness_values_array))], fitness_values_array)
    plt.rc('font', size=30)
    plt.tick_params(labelsize=18)
    plt.xlabel("Поколение", fontsize = 22)
    plt.ylabel("Фитнес-функция", fontsize = 22)
    plt.savefig('result/generations-fitness.png')


# -----------------------------------------------------------
# Сохраняет итоговое изображение
# -----------------------------------------------------------
def save_improved_image(improved_image, initial_image):
    plt.rc('font', size=12)
    plt.tick_params(labelsize=18)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.suptitle('Ядро улучшения (Локальный подход)')

    ax1.imshow(initial_image, cmap='gray', vmin=0, vmax=255)
    ax1.set_title("Исходое изображение")

    ax2.imshow(improved_image, cmap='gray', vmin=0, vmax=255)
    ax2.set_title("Улучшенное изображение")

    plt.savefig('result/improved.png')


def calculate_metrics(initial, improved):
    E_init, pix_count_init = sum_intensity(initial)
    LQ_init = level_of_adaptation(initial)
    entropy_init = measure_of_entropy(initial)
    print("Исходное изображение")
    print("Уровень адаптации по яркости", LQ_init)
    print("Количество краевых пикселей", pix_count_init)
    print("Суммарная интенсивность краевых писелей", E_init)
    print("Мера энтропии", entropy_init)
    print("---------------")
    E_imp, pix_count_imp = sum_intensity(improved)
    LQ_imp = level_of_adaptation(improved)
    entropy_imp = measure_of_entropy(improved)
    print("Улучшенное изображение")
    print("Уровень адаптации по яркости", LQ_imp)
    print("Количество краевых пикселей", pix_count_imp)
    print("Суммарная интенсивность краевых писелей", E_imp)
    print("Мера энтропии", entropy_imp)
