# ===== PARTE 1: WHITE PATCH =====

# ===== LIBRERÍAS =====
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ===== ALGORITMO WHITE PATCH =====
def white_patch_algorithm(img):
    """
    Corrige el balance de blancos usando el píxel más brillante.
    """
    img = img.astype(np.float32)
    max_vals = np.max(img, axis=(0, 1))  # máximo de cada canal
    for c in range(3):
        if max_vals[c] > 0:
            img[:, :, c] = img[:, :, c] / max_vals[c] * 255
    return np.clip(img, 0, 255).astype(np.uint8)

# ===== RUTA DE LA CARPETA (misma del Notebook) =====
carpeta = "white_patch"

# Crear carpeta de salida si no existe
carpeta_out = os.path.join(carpeta, "wp_resultados")
os.makedirs(carpeta_out, exist_ok=True)

# ===== PROCESAR TODAS LAS IMÁGENES =====
for archivo in os.listdir(carpeta):
    if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
        path_in = os.path.join(carpeta, archivo)

        # Leer imagen
        img = cv2.imread(path_in)
        if img is None:
            print(f"⚠️ No se pudo leer: {archivo}")
            continue

        # Aplicar algoritmo
        img_wp = white_patch_algorithm(img)

        # Guardar resultado
        nombre, ext = os.path.splitext(archivo)
        path_out = os.path.join(carpeta_out, f"{nombre}_wp{ext}")
        cv2.imwrite(path_out, img_wp)

        # Mostrar original y corregida
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_wp_rgb = cv2.cvtColor(img_wp, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(img_rgb)
        plt.title(f"Original: {archivo}")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(img_wp_rgb)
        plt.title("White Patch")
        plt.axis("off")
        plt.show()

        print(f"✅ Procesada {archivo} -> {nombre}_wp{ext}")

# ===== PARTE 2: HISTOGRAMAS =====

# ===== RUTAS RELATIVAS =====
img1_path = "white_patch/img1_tp.png"
img2_path = "white_patch/img2_tp.png"

# ===== CARGAR IMÁGENES EN ESCALA DE GRISES =====
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("⚠️ ¡ERROR! No se encontraron img1_tp.png y/o img2_tp.png")
else:
    # Nos aseguramos que tengan el mismo tamaño
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # =================== Mostrar imágenes ===================
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img1, cmap="gray")
    plt.title("Imagen 1")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img2, cmap="gray")
    plt.title("Imagen 2")
    plt.axis("off")
    plt.show()

    # =================== Calcular histogramas ===================
    bins = 32
    hist1 = cv2.calcHist([img1], [0], None, [bins], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [bins], [0, 256])

    # Normalizar para que las curvas sean comparables
    hist1 = hist1 / hist1.max()
    hist2 = hist2 / hist2.max()

    # =================== Graficar histogramas por separado ===================
    plt.figure(figsize=(8,4))
    plt.plot(hist1.ravel(), color='blue', linewidth=2, marker='o')
    plt.title("Histograma Imagen 1")
    plt.xlabel("Nivel de gris (agrupado en bins)")
    plt.ylabel("Frecuencia normalizada")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(hist2.ravel(), color='orange', linewidth=2, marker='o')
    plt.title("Histograma Imagen 2")
    plt.xlabel("Nivel de gris (agrupado en bins)")
    plt.ylabel("Frecuencia normalizada")
    plt.grid(True)
    plt.show()

    # =================== Comparación de histogramas ===================
    dist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    dist_chi = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    dist_bhatt = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    print()
    print("Correlación:", dist_corr)
    print("Chi-cuadrado:", dist_chi)
    print("Bhattacharyya:", dist_bhatt)

    # =================== Diferencia pixel a pixel ===================
    diff = cv2.absdiff(img1, img2)
    print("Suma de las diferencias de intensidad de todos los píxeles:", np.sum(diff))
    print()

    # Mostrar mapa de diferencias
    plt.figure(figsize=(8,6))
    plt.imshow(diff, cmap='hot')
    plt.colorbar(label='Diferencia de intensidad')
    plt.title("Mapa de diferencias entre imágenes")
    plt.show()