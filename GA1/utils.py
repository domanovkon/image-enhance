import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Строит график зависимости поколения от значения
# фитнес-функции
# -----------------------------------------------------------
def plot_generations_graph(fitness_values_array):
    plt.figure(figsize=(10, 10))
    plt.plot([i for i in range(len(fitness_values_array))], fitness_values_array)
    plt.title("Зависимость")
    plt.xlabel("Поколение")
    plt.ylabel("Фитнес-функция")
    plt.savefig('result/generations-fitness.png')