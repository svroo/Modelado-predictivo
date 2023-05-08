import numpy as np
import matplotlib.pyplot as plt


# Función para clasificar los puntos
def simplified_smo(
    C: float, tol: float, X: np.array, y: np.array, max_passes: int = 100
) -> np.array:
    """Algoritmo de Support Vector Machine simplificado.

    Args:
        C (float): Parametro de regularización (0.1, 1, 10, 100, etc.)
        tol (float): Valor de tolerancia (0.001, 0.0001)
        X (np.array): valores que queremos separar
        y (np.array): Etiqueta de los valores de x
        max_passes (int, optional): Número de veces que repite el algoritmo. Defaults to 100.

    Returns:
        alpha : valores de lagrange que es la solucion
        b : Umbral para la solución o intersección entre datos
    """
    # Inicializamos m, n para no hacerlo despues
    m, n = X.shape
    # Creamos un arreglo donde se almacenan los valores de alpha
    alpha = np.zeros(m)
    b = 0
    passes = 0

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            # Calculo del ei y el kernel al mismo tiempo usando el producto punto
            E_i = np.dot(alpha * y, np.dot(X, X[i])) + b - y[i]
            # Verificamos la condición de KKT (Kernel Karush-Kuhn-Tucker)
            if (y[i] * E_i < -tol and alpha[i] < C) or (
                y[i] * E_i > tol and alpha[i] > 0
            ):
                # Generamos un indice aleatorio
                j = np.random.choice(np.delete(np.arange(m), i))
                # calculamos el e_j
                E_j = np.dot(alpha * y, np.dot(X, X[j])) + b - y[j]
                # guardamos el valor antiguo de los alpha
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                # Calcular los límites de alpha[j] para que satisfagan las restricciones
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                # Si L = H, no se puede hacer nada
                if L == H:
                    continue
                # Calcular el valor no restringido de alpha[j]
                eta = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
                if eta >= 0:
                    continue
                # Calcular y recortar el nuevo valor de alpha[j]
                alpha[j] = alpha[j] - (y[j] * (E_i - E_j)) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                # Calcular los valores de b para los nuevos valores de alpha[i] y alpha[j]
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])
                b_i = (
                    b
                    - E_i
                    - y[i] * (alpha[i] - alpha_i_old) * np.dot(X[i], X[i])
                    - y[j] * (alpha[j] - alpha_j_old) * np.dot(X[i], X[j])
                )
                b_j = (
                    b
                    - E_j
                    - y[i] * (alpha[i] - alpha_i_old) * np.dot(X[i], X[j])
                    - y[j] * (alpha[j] - alpha_j_old) * np.dot(X[j], X[j])
                )
                if 0 < alpha[i] < C:
                    b = b_i
                elif 0 < alpha[j] < C:
                    b = b_j
                else:
                    b = (b_i + b_j) / 2
                num_changed_alphas += 1
        # Si no se hacen cambios en el ciclo, salimos del while
        # Si no, aumentamos el contador de pasos
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alpha, b
