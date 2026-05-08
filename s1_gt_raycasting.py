"""
========================================================================================
Entrada: Mallas (.obj, .ply, .stl)
Proceso: 
    1. Sampling (8192 puntos + Normales)
    2. Evaluación Gravitacional (Distancia al CoM)
    3. Evaluación Geométrica Local (PCA con KDTree)
    4. Evaluación de Agarre Externo/Interno (Alineación de Normales)
    5. Filtro cinemático (Volumetric Raycasting para Stroke < 10cm)
Salida: 
    - Matrices .npy (8192, 7) listas para entrenar la PointNet.
    - Archivos .ply con el mapa de calidad de agarre para verificar en MeshLab.
========================================================================================
"""

import os
import glob
import numpy as np
import trimesh
import pymeshlab as ml
from tqdm import tqdm
from scipy.spatial import KDTree

# CONFIGURACIÓN
INPUT_FOLDER = r"./Dataset/Mesh"        # Carpeta con mallas 3D originales
OUTPUT_FOLDER_NPY = r"./Dataset/Train"  # Salida para la red neuronal
OUTPUT_FOLDER_PLY = r"./Dataset/Visual" # Salida visual (MeshLab)
TARGET_POINTS = 8192                    # tamaño de la nube de puntos
MAX_GRIP_WIDTH = 0.1    # 10 cm. Si la pieza es más ancha de esto, se rechaza.

# PESOS DE LAS MÉTRICAS
W_GRAVITY = 0.3
W_GEOMETRY = 0.4
W_NORMAL = 0.3

def process_single_mesh(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0] # obtener un archivo de la lista
    try:
        # =====================================================================
        # 1. CARGA Y PREPROCESADO
        # =====================================================================

        # Lo cargamos y compruebamos que sea un mesh
        # WARNING: al forzar el mesh, si hay varios objetos en el archivo, los concatena. Si no lo usamos, crearía una escena
        mesh = trimesh.load(file_path, force='mesh') 
        
        # Si no es un mesh, no podemos calcular el raycasting
        if len(mesh.faces) == 0:
            return False, "El archivo es una Nube de Puntos, no un mesh. El Raycasting es imposible."
        
        # con fix_normals corregimos la dirección de las normales
            # da la vuelta a las que estén al revés (respecto a sus vecinos)
            # comprueba que las normales apunten hacia fuera
        mesh.fix_normals() # WARNING: si el mesh no está cerrado puede dar problemas

        # Generamos 8192 puntos en la superficie y obtener sus normales
        points, face_indices = trimesh.sample.sample_surface(mesh, TARGET_POINTS)

        # Centrado dinámico en memoria. hay que centrar los datos porque pointnet es sensible al desplazamiento. 
            # es mejor hacer el centrado en memoria que guardarlo en un archivo y cargarlo de nuevo
        centroide = np.mean(points, axis=0)
        points = points - centroide

        normals = mesh.face_normals[face_indices] # devuelve array (N_caras, 3) con el vector normal unitario de cada cara 
        
        # =====================================================================
        # 2. CÁLCULO DE MÉTRICAS (Gravedad, Geometría, Normales)
        # =====================================================================

        num_points = points.shape[0]

        # Calculamos el centro de masa (CoM) (promedio de todos los puntos) 
        com = np.mean(points, axis=0)
        
        # -------------------- A. Gravedad --------------------
        dist_vec = points - com # vector desde el CoM hasta cada punto (N, 3)
        distances = np.linalg.norm(dist_vec, axis=1) # usando Pitágoras, transformamos el vector en una distancia
        
        # Obtenemos la mayor y menor distancia (la usamos para normalizar)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        rango_dist = max_dist - min_dist
        if rango_dist < 1e-5: # Si la pieza es una esfera perfecta, evitamos dividir por 0
            rango_dist = 1.0
            
        # Normalizamos: La zona de la superficie más cerca al CoM saca 1.0, la más lejana 0.0
        """
        ((distances - min_dist) / rango_dist)
        # punto + cercano  -> (1.0 - 1.0) / 3.0 = 0.0
        # punto + lejano   -> (4.0 - 1.0) / 3.0 = 1.0

        #  1.0 - resultado 
        # punto + cercano  -> 1.0 - 0.0 = 1.0  -> score alto
        # punto + lejano   -> 1.0 - 1.0 = 0.0  -> score bajo
        """
        score_gravity = 1.0 - ((distances - min_dist) / rango_dist)
        
        # -------------------- B. Geometría Local (KDTree & PCA) --------------------
        tree = KDTree(points) # Para buscar de forma eficiente los vecinos
        indices = tree.query_ball_point(points, r=0.025) # Obtenemos todos los vecinos en ese radio (devuelve una lista de listas (indices y vecinos))
        
        score_geometry = np.zeros(num_points)
        for i, idx_vecinos in enumerate(indices):
            if len(idx_vecinos) < 5: continue # si tiene pocos vecinos, no va a ser un buen agarre
                
            local_cloud = points[idx_vecinos] # Nube local del punto i
            # Movemos la nube local al origen (restando su media). Esto es un requisito estricto 
            # para el PCA: queremos analizar solo la *forma* de la superficie, sin que importe 
            # en qué parte del espacio global está el objeto.
            local_centered = local_cloud - np.mean(local_cloud, axis=0) 
            
            # Matriz de covarianza y extracción de autovalores. Básicamente, esto nos dice 
            # cómo se distribuyen los puntos en 3D y en qué direcciones hay más "dispersión".
            cov = np.dot(local_centered.T, local_centered) / len(idx_vecinos)
            eigenvalues, _ = np.linalg.eigh(cov)
            # l1 es la dirección más estrecha (el grosor), l3 la más alargada.
            l1, l2, l3 = eigenvalues # l1 es la dirección más estrecha (el grosor), l3 la más alargada.
            
            # Metemos un épsilon (1e-9) para evitar que el programa crashee por 
            # una división por cero si todos los puntos fueran exactamente iguales.
            l_sum = l1 + l2 + l3 + 1e-9 
            # Extraemos las características geométricas basadas en cómo se relacionan las dimensiones:
            linearity = (l3 - l2) / l_sum # ¿Predomina 1D? -> Es un borde o arista.
            planarity = (l2 - l1) / l_sum # ¿Predomina 2D? -> Es una superficie plana.
            sphericity = l1 / l_sum # ¿Son parecidas? -> Es una esquina, bola o curva.
            
            # Calculamos la nota heurística del agarre. La pinza del Kinova funciona muy bien 
            # agarrando bordes (2.0) o zonas redondas (1.5). 
            # Malo para agarrar una superficie  plana, por eso la penalizamos (0.1)
            local_score = (linearity * 2.0) + (sphericity * 1.5) + (planarity * 0.1)
            
            # Convexidad/Concavidad
            # Para esto, comparamos hacia dónde apunta la normal de la superficie respecto al centro de masas (COM) del objeto.
            vec_to_center = com - points[i]
            alignment = np.dot(normals[i], vec_to_center)

            # Si la normal y el vector al centro apuntan "hacia el mismo lado" (producto escalar > 0), 
            # significa que estamos en una zona cóncava (ej: cuenco). 
            # Le damos mayor puntuación porque esos agarres patinan menos.
            if alignment > 0:
                local_score *= 1.5 # Premia agarre interior
                
            # Normalizamos entre [0-1]
            score_geometry[i] = np.clip(local_score, 0, 1)

        # -- C. Agarre Externo (Normales) --
        # Primero calculamos la longitud (norma) de los vectores que van desde el eje/centro 
        # del objeto hasta cada punto de la superficie.
        norms_dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
        # [WARNING] si un punto está exactamente en el centro, su longitud es 0.
        # Le ponemos un 1 temporal para evitar dividir entre cero.
        norms_dist[norms_dist == 0] = 1 
        # Dividimos el vector por su longitud. Así obtenemos un vector de longitud 1. 
        # Solo nos interesa la dirección.
        radial_vecs = dist_vec / norms_dist 
        
        # Hacemos el producto escalar entre la normal de la superficie y este vector radial.
        # ASí sabemos si la superficie está mirando directamente hacia afuera desde el centro del objeto?
        normal_alignment = np.sum(normals * radial_vecs, axis=1)
        # Nos quedamos con el valor absoluto. Si el valor es cercano a 1, significa que la 
        # superficie es casi perpendicular al centro (ideal para que la pinza entre recta 
        # sin resbalar). Si es cercano a 0, la superficie está muy inclinada y la pinza va a desplazar el objeto al cerrar.
        score_normal = np.abs(normal_alignment) 

        # -- CÁLCULO DE PUNTUACIÓN --
        # Combinamos las tres heurísticas que tenemos hasta ahora (gravedad, geometría y normales)
        base_score = (score_gravity * W_GRAVITY) + (score_geometry * W_GEOMETRY) + (score_normal * W_NORMAL)
        base_score = np.clip(base_score, 0, 1) # Normalizamos entre [0-1]

        # =====================================================================
        # 3. EL FILTRO CINEMÁTICO: RAYCASTING BIDIRECCIONAL
        # =====================================================================
        # Épsilon es un pequeño desplazamiento hacia dentro del objeto. Esto se debe a que si 
        # lanzamos un rayo láser exactamente desde la superficie del objeto, detectará un choque 
        # instantáneo con la propia superficie
        epsilon = 1e-4
        
        # Preparamos los rayos hacia ADENTRO del objeto (origen un poco metido, dirección -Normal)
        ray_origins_in = points - normals * epsilon
        ray_directions_in = -normals 
        
        # [WARNING] Lo mismo pero hacia afuera (vamos a hacer esto en el caso en el que un objeto tenga las normales invertidas)
        ray_origins_out = points + normals * epsilon
        ray_directions_out = normals 
        
        thickness = np.zeros(num_points) # para guardar el grosor en cada punto
        
        for i in range(num_points):
            origen_in = ray_origins_in[i]
            dir_in = ray_directions_in[i]
            
            # --- Disparo (-N) ---  
            # Lanzamos el rayo hacia adentro
            loc_in, _, _ = mesh.ray.intersects_location(
                ray_origins=[origen_in],
                ray_directions=[dir_in]
            )
            
            dist_in = 0.0
            # Si ha chocado con algo
            if len(loc_in) > 0:
                # Ponemos las coordenadas de los impactos en (x,y,z)
                loc_in = np.reshape(loc_in, (-1, 3))
                # Calculamos la distancia desde el origen del rayo hasta el impacto
                distancias = np.linalg.norm(loc_in - origen_in, axis=1)
                
                # Descartamos choques absurdamente cerca (posibles errores de la malla).
                dist_validas = distancias[distancias > 1e-4]


                if len(dist_validas) > 0:
                    # De los impactos válidos, nos quedamos con el más cercano (np.min).
                    dist_in = np.min(dist_validas)
                

            # --- Disparo (+N) ---
            # Lanzamos el rayo hacia afera (como la normal)
            origen_out = ray_origins_out[i]
            dir_out = ray_directions_out[i]
            
            # Corregido el orden de retorno también aquí
            loc_out, _, _ = mesh.ray.intersects_location(
                ray_origins=[origen_out],
                ray_directions=[dir_out]
            )
            
            dist_out = 0.0
            if len(loc_out) > 0:
                # Reorganizamos los puntos de impacto en 3D
                loc_out = np.reshape(loc_out, (-1, 3))
                # Calculamos las distancias de todos los impactos
                distancias = np.linalg.norm(loc_out - origen_out, axis=1)
                # Volvemos a ignorar la propia "piel" del objeto
                dist_validas = distancias[distancias > 1e-4]

                
                if len(dist_validas) > 0:
                    dist_out = np.min(dist_validas)
                
            # Recopilamos únicamente los rayos que han chocado contra algo físico.
            # Dejamos las distancias 0.0 (infinito)
            distancias_validas = []
            if dist_in > 0.0: distancias_validas.append(dist_in)
            if dist_out > 0.0: distancias_validas.append(dist_out)
            
            if distancias_validas:
                # la pinza debe rodear SIEMPRE la distancia más corta (el mínimo). Esto evita falsos rechazos en 
                # piezas cóncavas o huecas (ej. el interior de un cilindro o cuenco).
                thickness[i] = min(distancias_validas)
            else:
                # Si ningún rayo choca con nada, es una malla abierta o un plano infinito.
                # Ponemos un grosor infinito para que sea descartado por la máscara MAX_GRIP_WIDTH.
                thickness[i] = float('inf')
            
        print(f"\n[DEBUG] {file_name}")
        print(f"Grosor Medio: {np.mean(thickness):.4f} m | Máximo: {np.max(thickness):.4f} m")
        # Calculamos qué porcentaje de los puntos tienen un grosor menor o igual a lo que 
        # puede abrir la pinza del robot (MAX_GRIP_WIDTH).
        porcentaje_validos = (np.sum(thickness <= MAX_GRIP_WIDTH) / num_points) * 100
        print(f"Puntos que CABEN (<= {MAX_GRIP_WIDTH*100}cm): {porcentaje_validos:.1f}%")

        # Si el grosor es <= 10cm, es agarrable.
        valid_grip_mask = thickness <= MAX_GRIP_WIDTH
        # Multiplicamos la nota por la máscara. Los puntos donde no cabe la pinza (False = 0) 
        # tendrán una puntuación de 0. Los demás mantendrán su nota.
        final_score = base_score * valid_grip_mask

        # =====================================================================
        # 4. GUARDADO DE ARCHIVOS
        # =====================================================================
        
        # --- NORMALIZACIÓN A ESFERA UNITARIA (para la red neuronal) ---
        # Calculamos el punto más lejano al centro (radio máximo)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        
        if max_dist < 1e-6: max_dist = 1.0
            
        # Dividimos todas las coordenadas por ese radio (encogemos la pieza a radio 1.0)
        points_normalized = points / max_dist

        # A) Formato .npy (Para la Red Neuronal) -> USAMOS LOS PUNTOS NORMALIZADOS
        labeled_data = np.hstack([points_normalized, normals, final_score.reshape(-1, 1)]).astype(np.float32)
        np.save(os.path.join(OUTPUT_FOLDER_NPY, f"{file_name}.npy"), labeled_data)
        
        # B) Formato .ply (Para visualizar el Heatmap en MeshLab)
        visual_mesh = ml.Mesh(
            vertex_matrix=points.astype(np.float64),
            v_normals_matrix=normals.astype(np.float64),
            v_scalar_array=final_score.astype(np.float64) # El score se guarda como 'Quality'
        )
        
        mset = ml.MeshSet()  
        mset.add_mesh(visual_mesh)
        mset.save_current_mesh(os.path.join(OUTPUT_FOLDER_PLY, f"{file_name}_VISUAL.ply"),
                               save_vertex_color=False,
                               save_vertex_quality=True,
                               binary=True)
        
        return True, ""
        
    except Exception as e:
        return False, str(e)


def main():
    # Crear carpetas si no existen
    for folder in [OUTPUT_FOLDER_NPY, OUTPUT_FOLDER_PLY]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    # Buscamos .obj, .ply y .stl de una sola pasada sumando las listas
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.obj")) + \
            glob.glob(os.path.join(INPUT_FOLDER, "*.ply")) + \
            glob.glob(os.path.join(INPUT_FOLDER, "*.stl"))
    
    if len(files) == 0:
        print(f" [AVISO] No se encontraron mallas 3D en {INPUT_FOLDER}")
        return
        
    print(f" [INFO] Iniciando Pipeline End-to-End para {len(files)} piezas...")
    
    errores = 0
    for f in tqdm(files, desc="Procesando Ground Truth"):  
        ok, msg = process_single_mesh(f)
        if not ok:
            print(f"\n[ERROR] {os.path.basename(f)}: {msg}")
            errores += 1
            
    print(f"\n Pipeline completado. Éxitos: {len(files)-errores} | Errores: {errores}")
    print(f" Matrices de entrenamiento guardadas en: {OUTPUT_FOLDER_NPY}")

if __name__ == "__main__":
    main()
