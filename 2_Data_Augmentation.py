"""
===============================================
DATA AUGMENTATION
===============================================
Multiplica el dataset x4 aplicando:
1. Rotaciones aleatorias (Ejes X, Y y Z simultáneos)
2. Ruido Gaussiano (Jittering) de +/- 2mm a las coordenadas X, Y, Z
Asegura que las normales giran con los puntos
===============================================
"""

import os
import glob
import numpy as np
from tqdm import tqdm

# --- CONFIGURACIÓN ---
INPUT_FOLDER = r"./Dataset/Train"             # Carpeta con tus 120 archivos .npy originales
OUTPUT_FOLDER = r"./Dataset/Train_Augmented"  # Carpeta nueva donde habrá 480 archivos
# Cuántas copias extra generar por cada pieza original
NUM_AUMENTOS = 3    # Esto evita que la red neuronal "memorice" (overfitting) la posición exacta en la que escaneaste el objeto.

# ruido (en metros). Movimiento de 2mm, hasta a un máximo de 5mm en la posición de los puntos.
NOISE_STD = 0.002
NOISE_CLIP = 0.005

def rotar_3d(points, normals):
    """
    Genera ángulos aleatorios para X, Y y Z, crea sus matrices de rotación,
    las combina y las aplica al objeto y a sus normales.
    """
    # Ángulos aleatorios (0 a 360 grados) para cada eje
    ax = np.random.uniform(0, 2 * np.pi)
    ay = np.random.uniform(0, 2 * np.pi)
    az = np.random.uniform(0, 2 * np.pi)
    
    # Matriz de rotación en X (Pitch). Gira hacia adelante o atrás
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax),  np.cos(ax)]
    ])
    
    # Matriz de rotación en Y (Yaw). Gira sobre si mismo
    R_y = np.array([
        [ np.cos(ay), 0, np.sin(ay)],
        [ 0,          1, 0],
        [-np.sin(ay), 0, np.cos(ay)]
    ])
    
    # Matriz de rotación en Z (Roll). Gira como un reloj
    R_z = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az),  np.cos(az), 0],
        [0,           0,          1]
    ])
    
    # Matriz de rotación combinada (R = Rz * Ry * Rx)
    R_combinada = np.dot(R_z, np.dot(R_y, R_x))
    
    # Multiplicamos la lista de puntos (N, 3) por la matriz transpuesta (3, 3)
    # Es necesario realizar la transpuesta para que coincidan las dimensiones 
    points_rot = np.dot(points, R_combinada.T)
    # WARNING si rotamos los puntos, hay que rotar las normales
    normals_rot = np.dot(normals, R_combinada.T)
    
    return points_rot, normals_rot

def aplicar_ruido(points):
    """
    Añade un ruido microscópico a los puntos para evitar el sobreajuste.
    """
    # Generamos una matriz de ruido aleatorio del mismo tamaño que la nube de puntos (distribucion gaussiana)
    ruido = np.random.normal(0, NOISE_STD, points.shape)
    # aunque es gaussiano, vamos a poner un límite al error que pase de los 5mm (+/- 0.005m)
    ruido = np.clip(ruido, -NOISE_CLIP, NOISE_CLIP) 
    
    return points + ruido

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.npy"))
    
    if len(files) == 0:
        print(f"[AVISO] No se encontraron archivos .npy en {INPUT_FOLDER}")
        return
        
    print(f"[INFO] Iniciando Data Augmentation: {len(files)} piezas -> {len(files) * (NUM_AUMENTOS + 1)} piezas.")
    
    for file_path in tqdm(files):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Cargamos los datos
        data = np.load(file_path)
        points = data[:, :3]
        normals = data[:, 3:6]
        scores = data[:, 6:] 
        
        # Guardamos el archivo original
        np.save(os.path.join(OUTPUT_FOLDER, f"{file_name}_orig.npy"), data)
        
        # Generamos las copias
        for i in range(NUM_AUMENTOS):
            # Rotamos puntos y normales
            pts_rot, norms_rot = rotar_3d(points, normals)
            # Añadimos ruido SOLO a los puntos.
            # Las normales no, porque sumar ruido destruiría su longitud unitaria (dejando de ser vectores) 
            # y desviaría sus ángulos al azar, arruinando la matemática de los agarres.
            pts_final = aplicar_ruido(pts_rot)
            
            # Ensamblamos el nuevo tensor
            data_aug = np.hstack([pts_final, norms_rot, scores]).astype(np.float32)
            
            # Guardar. Hay que tener cuidado de no sobreescribir con el mismo nombre
            np.save(os.path.join(OUTPUT_FOLDER, f"{file_name}_aug_{i+1}.npy"), data_aug)

    print(f"\n[OK] Dataset aumentado guardado en: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
