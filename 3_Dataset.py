"""
Dataset de PyTorch para cargar las nubes de puntos preprocesadas.
Lee directamente los archivos .npy generados por el Ground Truth.
"""
import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np


"""
hacemos porque si cargamos todos los datos a la vez en la memoria RAM, colapsaría. 
Heredar de `Dataset` de PyTorch nos permite hacer una carga perezosa.
Solo leemos el archivo del disco duro cuando la red neuronal lo necesita para entrenar
En vez de cargar el dataset completo, creamos una lista con los nombres (ocupa mucho menos) y los vamos obteniendo de uno en uno
"""
class SatelliteDataset(Dataset):
    def __init__(self, root_dir):
        
        # Guardamos la ruta de los archivos, no los datos en sí.
        self.root_dir = root_dir
        
        # Buscamos todos los archivos .npy en la carpeta (N, 7)
        # glob.glob devuelve una lista de strings con las rutas completas de todos 
        # los archivos. Esta lista es lo único que mantenemos vivo en la RAM (ocupa muy poco).
        self.file_list = glob.glob(os.path.join(root_dir, "*.npy")) 
        
        # Comprobamos que hay alguno
        if len(self.file_list) == 0:
            print(f"[ERROR]: No se encontraron archivos .npy en {root_dir}")
        else:
            print(f"[INFO]: Dataset listo: {len(self.file_list)} archivos encontrados.")


    # PyTorch necesita saber el tamaño total del dataset para dividir en epocas y lotes
    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        """
        # El DataLoader le pide un índice y este método tiene que ir al disco duro,
        #  leerla y prepararla para la GPU.
        """
        try:
            # La ruta del archivo a buscar
            file_path = self.file_list[idx]
            
            # data tiene forma (N, 7) -> [X, Y, Z, Normal_X, Normal_Y, Normal_Z, Puntuacion_Agarre]
            data = np.load(file_path)
            
            # Separar las características
            points = data[:, 0:3]
            normals = data[:, 3:6]
            scores = data[:, 6] # Ground Truth

            # --- INVARIANCIA A LA TRASLACIÓN ---
            # Si a la IA le pasas una taza en la coordenada (0,0,0) y otra igual en (10,10,10), 
            # se pensará que son objetos distintos y no los agarrará igual
            # Calculamos el centroide (la media de todas las X, Y, Z) y se lo restamos a cada punto.
            # para centrar todos los objetos al origen (0,0,0).
            centroid = np.mean(points, axis=0)
            points_centered = points - centroid

            # Normalmente, en redes PointNet, aquí se dividirían todos los puntos por el más alejado 
            # para encajar el objeto en una esfera perfecta de tamaño [-1, 1].
            # ¿Por qué NO lo hacemos aquí? 
            # Porque si encogemos un coche y agrandamos un dado para que midan lo mismo, la IA 
            # pensará que ambos miden lo mismo. 
            # En Grasping, el tamaño absoluto (en metros) es crítica y necesaria. Si un punto está 
            # (No hay problema en usar números crudos porque ya son pequeños).

            # Pegamos la geometría (X,Y,Z) con las direcciones de las superficies (Nx,Ny,Nz).
            # Ahora tenemos una matriz (N, 6)
            network_input = np.hstack((points_centered, normals))

            # Convertir a Tensores (.float() para compatibilidad con la GPU)
            input_tensor = torch.from_numpy(network_input).float()  # entrada a la red 
            target_tensor = torch.from_numpy(scores).float()        # GT
            centroid_tensor = torch.from_numpy(centroid).float()    # centroid

            # Requisito de pointnet: las redes basadas en PointNet necesitan los datos como (Canales, Puntos)
            # en lugar de (Puntos, Canales), 
            # Hacemos una transposición: pasamos de matriz (Nx6) a (6xN).
            input_tensor = input_tensor.transpose(0, 1)
            
            # Lo devolvemos todo, el DataLoader juntará varios de estos packs para crear un Batch 
            return input_tensor, target_tensor, centroid_tensor

        except Exception as e:
                    # Tolerancia a fallos: Si un archivo .npy está corrupto o a medias,
                    # lo notificamos y cargamos recursivamente el siguiente índice.
                    # Esto evita que todo el entrenamiento falle por 1 archivo.
                    print(f"[WARNING] Error cargando el índice {idx} ({file_path}): {e}")
                    new_idx = (idx + 1) % len(self.file_list)
                    return self.__getitem__(new_idx)
        
 
