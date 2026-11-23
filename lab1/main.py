
from pathlib import Path
import numpy as np

data = np.loadtxt(
    Path(__file__).with_name("route_points.txt"),
    delimiter="|",
    skiprows=1
)

lat = data[:, 1]
lon = data[:, 2]
y = data[:, 3]

R = 6371000

phi1 = np.radians(lat[:-1])
phi2 = np.radians(lat[1:])
dphi = phi2 - phi1
dlambda = np.radians(np.diff(lon))

a = (np.sin(dphi / 2)**2
     + np.cos(phi1) * np.cos(phi2)
     * np.sin(dlambda / 2)**2)

steps = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
x = np.concatenate(([0.0], np.cumsum(steps)))


def thomas(lower, diagonal, upper, rhs):
    n = len(diagonal)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    solution = np.zeros(n)

    alpha[0] = upper[0] / diagonal[0]
    beta[0] = rhs[0] / diagonal[0]

    for i in range(1, n):
        denominator = diagonal[i] - lower[i] * alpha[i - 1]
        alpha[i] = upper[i] / denominator
        beta[i] = (rhs[i] - lower[i] * beta[i - 1]) / denominator

    solution[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        solution[i] = beta[i] - alpha[i] * solution[i + 1]

    return solution


def build_spline(x, y):
    n = len(x)
    h = np.diff(x)

    lower = np.zeros(n)
    diagonal = np.ones(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)

    for i in range(1, n - 1):
        lower[i] = h[i - 1]
        diagonal[i] = 2 * (h[i - 1] + h[i])
        upper[i] = h[i]

        rhs[i] = 3 * (
            (y[i + 1] - y[i]) / h[i]
            - (y[i] - y[i - 1]) / h[i - 1]
        )

    c = thomas(lower, diagonal, upper, rhs)

    a = y[:-1].copy()
    b = (np.diff(y) / h
         - h * (2 * c[:-1] + c[1:]) / 3)
    d = np.diff(c) / (3 * h)

    return a, b, c, d


def evaluate_spline(nodes, coefficients, points):
    a, b, c, d = coefficients

    indices = np.searchsorted(nodes, points, side="right") - 1
    indices = np.clip(indices, 0, len(nodes) - 2)

    dx = points - nodes[indices]

    return (a[indices]
            + b[indices] * dx
            + c[indices] * dx**2
            + d[indices] * dx**3)


coefficients = build_spline(x, y)
a, b, c, d = coefficients

print(f"Кількість GPS-точок: {len(x)}")
print(f"Довжина маршруту: {x[-1]:.2f} м")

print("\nКоефіцієнти кубічного сплайна:")
print("№ | a | b | c | d")

for i in range(len(x) - 1):
    print(
        f"{i + 1:2d} | {a[i]:.4f} | {b[i]:.6f} | "
        f"{c[i]:.8f} | {d[i]:.10f}"
    )


print("\nПохибки інтерполяції:")

for count in (10, 15, 20):
    indices = np.linspace(0, len(x) - 1, count, dtype=int)

    nodes = x[indices]
    values = y[indices]

    spline = build_spline(nodes, values)
    predicted = evaluate_spline(nodes, spline, x)

    errors = y - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(np.abs(errors))

    omitted = np.ones(len(x), dtype=bool)
    omitted[indices] = False
    omitted_mae = np.mean(np.abs(errors[omitted]))

    print(f"\n{count} вузлів:")
    print(f"MAE: {mae:.4f} м")
    print(f"RMSE: {rmse:.4f} м")
    print(f"Максимальна похибка: {max_error:.4f} м")
    print(f"MAE невикористаних точок: {omitted_mae:.4f} м")
