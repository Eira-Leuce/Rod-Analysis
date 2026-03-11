import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Параметры стержня
E = 206e9  # Модуль Юнга, Па
l = [3, 1, 2]  # Длины участков, м
A = [1e-6, 5e-6, 4e-6]  # Площади сечений, м^2
P = [0, 3e3, -2e3, -1e3]  # Точечные усилия, Н
q = [1e3, 0, 0]  # Распределенные нагрузки, Па/м


# Функция для расчета жесткости элемента стержня (линейный элемент)
def stiffness_matrix_linear(E, A, L):
    """Возвращает матрицу жесткости для линейного стержневого элемента."""
    return E * A / L * np.array([[1, -1], [-1, 1]])


# Функция для расчета жесткости кубического элемента
def stiffness_matrix_cubic(E, A, L):
    """Возвращает матрицу жесткости для кубического стержневого элемента."""
    return (E * A / (L ** 3)) * np.array([[7, -8, 3, -4],
                                          [-8, 16, -8, 8],
                                          [3, -8, 7, -4],
                                          [-4, 8, -4, 7]])


# Расчет методом конечных элементов (МКЭ)
def finite_element_method(E, A, l, P, q, num_elements, cubic_elements=False):
    n = num_elements  # Количество элементов
    total_length = sum(l)  # Общая длина стержня
    nodes = n + 1  # Количество узлов
    K = np.zeros((nodes, nodes))  # Общая матрица жесткости
    F = np.zeros(nodes)  # Вектор нагрузок

    # Разбиение на элементы
    element_length = total_length / n
    for i in range(n):
        # Длина и площадь сечения текущего элемента
        L = element_length
        A_current = A[i % len(A)]  # Площадь сечения для текущего элемента
        # Выбор матрицы жесткости для кубического или линейного элемента
        if cubic_elements and q[i] != 0:
            Ke = stiffness_matrix_cubic(E, A_current, L)
        else:
            Ke = stiffness_matrix_linear(E, A_current, L)

        # Индексы узлов для текущего элемента
        dof = [i, i + 1]
        # Вставляем матрицу жесткости в общую матрицу K
        K[dof[0], dof[0]] += Ke[0, 0]
        K[dof[0], dof[1]] += Ke[0, 1]
        K[dof[1], dof[0]] += Ke[1, 0]
        K[dof[1], dof[1]] += Ke[1, 1]

        # Добавление точечных нагрузок
        if i < len(P):
            F[i + 1] += P[i]

        # Добавление распределенной нагрузки (интегрирование по элементу)
        if i < len(q):
            # q[i] является распределенной нагрузкой на элементе
            F[dof] += q[i] * L / 2  # Простая аппроксимация для распределенной нагрузки

    # Учет граничных условий: например, задаем нулевое перемещение на первом узле
    K[0, 0] = 1e10  # Большая жесткость на первом узле (фиксированный узел)
    F[0] = 0  # Перемещение на первом узле равно нулю

    # Решение системы линейных уравнений для перемещений
    u = np.linalg.solve(K, F)

    return u, K, F


# 1. Разбиение на 3 КЭ с линейными стержневыми КЭ
u1, K1, F1 = finite_element_method(E, A, l, P, q, 3, cubic_elements=False)

# 2. Разбиение на 6 КЭ с равномерным шагом
u2, K2, F2 = finite_element_method(E, A, l, P, q, 6, cubic_elements=False)

# 3. Разбиение на 3 КЭ, с линейными КЭ для точечных нагрузок и кубическими для распределенных
u3, K3, F3 = finite_element_method(E, A, l, P, q, 3, cubic_elements=True)

# Для интерполяции перемещений на массив x_anal
x_anal = np.linspace(0, sum(l), 1000)

# Интерполяция перемещений для каждого случая
interp_u1 = interp1d(np.linspace(0, sum(l), len(u1)), u1, kind='linear', fill_value="extrapolate")
interp_u2 = interp1d(np.linspace(0, sum(l), len(u2)), u2, kind='linear', fill_value="extrapolate")
interp_u3 = interp1d(np.linspace(0, sum(l), len(u3)), u3, kind='linear', fill_value="extrapolate")

# Получаем значения перемещений на всех точках x_anal
u1_interp = interp_u1(x_anal)
u2_interp = interp_u2(x_anal)
u3_interp = interp_u3(x_anal)

# Построение графиков перемещений
plt.figure(figsize=(12, 8))

# График для 3 КЭ (линейные)
plt.subplot(3, 1, 1)
plt.plot(x_anal, u1_interp, label='u (3 КЭ, линейные)')
plt.title('Перемещения для 3 КЭ (линейные)')
plt.xlabel('x, м')
plt.ylabel('u, м')
plt.grid(True)
for i, txt in enumerate(u1_interp[::100]):  # Вывод значений перемещений через 100 точек
    plt.text(x_anal[i * 100], txt, f'{txt:.6f}', fontsize=8, color='blue')

# График для 6 КЭ (линейные)
plt.subplot(3, 1, 2)
plt.plot(x_anal, u2_interp, label='u (6 КЭ, линейные)')
plt.title('Перемещения для 6 КЭ (линейные)')
plt.xlabel('x, м')
plt.ylabel('u, м')
plt.grid(True)
for i, txt in enumerate(u2_interp[::100]):  # Вывод значений перемещений через 100 точек
    plt.text(x_anal[i * 100], txt, f'{txt:.6f}', fontsize=8, color='blue')

# График для 3 КЭ (линейные и кубические)
plt.subplot(3, 1, 3)
plt.plot(x_anal, u3_interp, label='u (3 КЭ, линейные + кубические)')
plt.title('Перемещения для 3 КЭ (линейные и кубические)')
plt.xlabel('x, м')
plt.ylabel('u, м')
plt.grid(True)
for i, txt in enumerate(u3_interp[::100]):  # Вывод значений перемещений через 100 точек
    plt.text(x_anal[i * 100], txt, f'{txt:.6f}', fontsize=8, color='blue')

plt.tight_layout()
plt.show()
