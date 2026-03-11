import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Параметры стержня
E = 206e9  # Модуль Юнга, Па (переход от ГПа к Па)
l = [3, 1, 2]  # Длины участков, м
A = [1e-6, 5e-6, 4e-6]  # Площади сечений, м^2
P = [0, 3e3, -2e3, -1e3]  # Точечные усилия, Н
q = [1e3, 0, 0]  # Распределенные нагрузки, Па/м

# Подготовка массивов для эпюр
x_analit = np.linspace(0, sum(l), 1000)  # Для построения эпюр
N_analit = np.zeros_like(x_analit)
sigma_analit = np.zeros_like(x_analit)
u_analit = np.zeros_like(x_analit)

# Реакция опоры (R)
R = sum(P) + sum(q[i] * l[i] for i in range(3))  # Уравнение равновесия
print(f"Реакция опоры R: {R:.2f} Н\n")

# Расчет эпюр на каждом участке
x_curr = 0  # Текущая координата начала участка
u_prev = 0  # Перемещение на предыдущем участке

for i in range(3):
    # Границы участка
    x_start = x_curr
    x_end = x_curr + l[i]

    # Массив x на текущем участке
    x_section = x_analit[(x_analit >= x_start) & (x_analit <= x_end)]

    # Суммарные распределенные нагрузки до текущего участка
    q_sum = sum(q[j] * l[j] for j in range(i))

    # Суммарные точечные усилия до текущего участка
    P_sum = sum(P[j] for j in range(1, i + 1))

    # Продольная сила N на текущем участке
    N_section = (
            R
            - q_sum  # Суммарные распределенные нагрузки до участка i
            - P_sum  # Точечные нагрузки до участка i
            - q[i] * (x_section - x_start)  # Распределенная нагрузка на текущем участке
    )
    N_analit[(x_analit >= x_start) & (x_analit <= x_end)] = N_section

    # Напряжение σ на текущем участке
    sigma_section = N_section / A[i]
    sigma_analit[(x_analit >= x_start) & (x_analit <= x_end)] = sigma_section

    # Перемещение на текущем участке
    u_section = (
            u_prev  # Перемещение с предыдущего участка
            + R * (x_section - x_start) / (E * A[i])  # От реакции опоры
            - q[i] * (x_section - x_start) ** 2 / (2 * E * A[i])  # От распределенной нагрузки
    )

    # Учет точечных нагрузок на предыдущих участках
    for j in range(1, i + 1):
        u_section -= P[j] * (x_section - (x_curr if j == i else sum(l[:j]))) / (
                    E * A[i])  # От точечных нагрузок

    u_analit[(x_analit >= x_start) & (x_analit <= x_end)] = u_section
    u_prev = u_section[-1]  # Обновляем начальное перемещение для следующего участка

    # Точные значения на границах участка
    a = x_start
    b = x_end
    N_a = N_section[0]
    N_b = N_section[-1]
    sigma_a = sigma_section[0]
    sigma_b = sigma_section[-1]
    u_a = u_section[0]
    u_b = u_section[-1]

    # Вывод в консоль
    print(f"Участок {i + 1} (от {a} до {b} м):")
    print(f"  Продольная сила N(a) = {N_a:.2f} Н, N(b) = {N_b:.2f} Н")
    print(f"  Напряжение σ(a) = {sigma_a:.2f} Па, σ(b) = {sigma_b:.2f} Па")

    x_curr = x_end  # Переход к следующему участку
    u_prev = 0  # Начальное перемещение
    x_curr = 0

    for i in range(len(l)):
        # Границы текущего участка
        x_start = x_curr
        x_end = x_curr + l[i]

        # Массив x на текущем участке
        x_section = x_analit[(x_analit >= x_start) & (x_analit <= x_end)]

        # Суммарные распределенные нагрузки до текущего участка
        q_sum = sum(q[j] * l[j] for j in range(i))

        # Суммарные точечные усилия до текущего участка
        P_sum = sum(P[j] for j in range(1, i + 1))

        # Продольная сила N на текущем участке
        N_section = (
                R
                - q_sum  # Суммарные распределенные нагрузки до участка i
                - P_sum  # Точечные нагрузки до участка i
                - q[i] * (x_section - x_start)  # Распределенная нагрузка на текущем участке
        )
        N_analit[(x_analit >= x_start) & (x_analit <= x_end)] = N_section


        # Функция для интегрирования
        def integrand(x):
            # Восстанавливаем N(x) для текущей точки x, используя линейную интерполяцию
            N_x = np.interp(x, x_section, N_section)
            return (N_x) / (E * A[i])


        # Вычисляем перемещения на текущем участке
        u_section = np.zeros_like(x_section)
        for j, x in enumerate(x_section):
            u_section[j] = u_prev + integrate.quad(integrand, x_start, x)[0]

        u_analit[(x_analit >= x_start) & (x_analit <= x_end)] = u_section
        u_prev = u_section[-1]
        x_curr = x_end

# Построение графиков
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(x_analit, N_analit, label='N(x)')
plt.title('Эпюра продольных сил')
plt.xlabel('x, м')
plt.ylabel('N, Н')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x_analit, sigma_analit, label='σ(x)', color='orange')
plt.title('Эпюра напряжений')
plt.xlabel('x, м')
plt.ylabel('σ, Па')
plt.grid(True)

plt.tight_layout()
plt.show()

# Построение графика перемещений с подписями точек
plt.figure(figsize=(10, 6))
plt.plot(x_analit, u_analit, label='u(x)')
plt.title('Эпюра перемещений (интегральный метод)')
plt.xlabel('x, м')
plt.ylabel('u, м')
plt.grid(True)
plt.legend()

# Находим индексы точек, соответствующих границам участков
x_points = np.cumsum(l)
x_points = np.insert(x_points, 0, 0)  # Добавляем начальную точку 0

# Добавляем подписи для каждой точки на графике
for x in x_points:
    u_value = np.interp(x, x_analit, u_analit) # находим значение u для точки x
    plt.plot(x, u_value, 'ro')  # 'ro' - красные кружки для выделения точек
    plt.text(x, u_value, f'({x:.1f}, {u_value:.6f})', ha='center', va='bottom')


plt.show()