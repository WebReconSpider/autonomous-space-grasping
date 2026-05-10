"""
Este script implementa el núcleo del modelo de aprendizaje profundo encargado de 
predecir la viabilidad de agarre sobre nubes de puntos 3D. Apoyándose en la 
arquitectura PointNet, la red neuronal se estructura en dos módulos principales:

- ENCODER: Analiza la nube de puntos y extrae un "vector latente" 
   que resume la geometría global de toda la pieza.
- DECODER: Coge esa visión global del objeto, la combina con la información 
   local exacta de cada punto (posición y normales), y emite un veredicto. 

El resultado final es un mapa de calor probabilístico (valores de 0 a 1 generados por 
una Sigmoide) donde cada punto de la pieza recibe una nota que indica la probabilidad 
de éxito si la pinza robótica intentara un agarre en esa coordenada exacta.
"""

import torch
from torch import nn

"""clase encoder que hereda de Module. Asi pytorch puede calcular gradientes, pesos y puedo usar la gpu"""
class Encoder(nn.Module):
    """
    La inicialización del encoder necesita:
        - input_channels: Cuántos datos hay por punto (6: x,y,z,nx,ny,nz)
        - latent_dim: El tamaño del vector latente (ej. 1024)
    """
    def __init__(self, input_channels, latent_dim): # hemos quitado los valores fijos que tenía input_channels, latent_dim ya que al usar optimización bayesiana, va a usar el tamaño "óptimo"
        super().__init__() # llama a nn.Module para que inicialice
        
        # Agregar secuencialmente las capas: Input (6) -> 64 -> 128 -> 1024 (Latente)
        # PyTorch inicializa los pesos de las capas nn.Conv1d automáticamente con valores aleatorios 
        # (normalmente usando un método llamado Kaiming/He initialization, que está 
        # matemáticamente optimizado para que las redes con ReLUs no se atasquen al principio).
        self.mlp_compartido = nn.Sequential( # MLP
            # Capa 1: Entrada 6 canales y salida 64 características
            nn.Conv1d(input_channels, 64, 1), # kernel_size=1 trata los puntos por separado, sin mezclarlos,  esto hace que "siempre haya las mismas filas 8192"
            nn.BatchNorm1d(64), # Normalización, ajusta los 64 valores para ue tengan una media cercana a 0 (ayuda a entrenar más rápido)
            nn.ReLU(),          # Función de activación (no linealidad) convierte los numeros negativos a 0 

            # Capa 2: De 64 a 128
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Capa 3: De 128 a tamaño del vector latente
            nn.Conv1d(128, latent_dim, 1),
            nn.BatchNorm1d(latent_dim)
        ) 
        """ 
        No se suele poner ReLU antes del Pooling para permitir que 
        el vector latente retenga características tanto positivas como negativas.
        """
            

    """Método forward define el calculo realizado en cada llamada. nos permite tomar un dato de entrada,
    procesarlo en la red y generar una salida
    cuando escribimos "salida = mi_modelo(mis_datos)", PyTorch internamente 
    llama a este def forward() y va registrando cada operación matemática en un grafo. 
    Así, cuando luego le pides calcular el error (Loss), sabe hacer el camino inverso (backpropagation).
    """
    def forward(self, x):
        # x es el dato de entrada, tiene forma: (Batch, 6, 8192)

        # EXTRAER LAS CARACTERÍSTICAS (MLP compartido)
        # ---------------------------------------------
        # La salida será: (Batch, 1024, 8192) -> Cada punto ahora tiene 1024 características
        x = self.mlp_compartido(x) 

        # GLOBAL MAX POOLING 
        # ---------------------------------------------
        """
         Tomamos el valor máximo de cada una de las 1024 características entre todos los puntos.
        Al quedarnos solo con el máximo, logramos que NO importe el orden de los puntos.
        El eje "2" es el que vamos a aplastar (Batch, Canales, Puntos <- este)

        
        torch.max devuelve una tupla (Valores, Índices)
            Los Valores son la intensidad máxima de la característica y los Indices son las posiciones donde están
            Ponemos [0] al final porque solo queremos los Valores, los índices no nos sirven.
        es necesario usar torch.max (y no otro como numpy) ya que no rompe el backpropagation, 
            permite el cálculo de gradientes, trabajar con GPU en paralelo y toma como entrada tensores"""
        vector_latente = torch.max(x, 2)[0] 
        
        # Salida: vector de tamaño (Batch, 1024)
        return vector_latente
    

    """ 
    Clase principal.
    No queremos que el decoder reconstruya el satélite, sino que clasifique lo bueno que es 
    cada uno de los puntos para el agarre
    """
class PointNetGrasping(nn.Module): 
    def __init__(self, input_channels=6, latent_dim=1024):
        super().__init__()

        # ENCODER
        # ---------------------------------
        # crear la instancia del Encoder
        self.encoder = Encoder(input_channels=input_channels, latent_dim=latent_dim)

        """
        Calculamos el tamaño de la entrada del decoder: 
        1024 del vector latente (contexto global) + 6 datos originales del punto (posición) -> 1030
        """
        self.decoder_input_dim = latent_dim + input_channels


        # El decoder va reduciendo la información hasta llegar a un único número (para cada uno de los puntos de la nube)
        self.decoder = nn.Sequential(
            # Capa 1: Bajamos de 1030 a 512 características
            nn.Conv1d(self.decoder_input_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            # Capa 2: Bajamos de 512 a 256
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Capa 3: Bajamos de 256 a 128
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Capa FINAL: De 128 a 1 SOLA SALIDA (El Score de Agarre)
            # No usamos BatchNorm al final para no alterar la predicción cruda.
            nn.Conv1d(128, 1, 1) 

        )

        self.sigmoid = nn.Sigmoid() # Ground Truth está estrictamente entre 0 y 1.



    def forward(self, x):
        # x tiene forma: (Batch, 6, 8192)
        # Guardamos una copia de la entrada original para la concatenación
        puntos_originales = x 


        # ENCODER
        # ---------------------------------------------------------
        # Obtenemos el vector global: (Batch, 1024)
        global_feat = self.encoder(x) 


        # EXPANSIÓN (Repetir el vector global)
        # ---------------------------------------------------------
        # Tenemos (Batch, 1024) y queremos (Batch, 1024, 8192)
        # Primero añadimos una dimensión extra al final: (Batch, 1024, 1)
        global_feat = global_feat.unsqueeze(2) 
        
        # Luego repetimos ese vector 8192 veces en el eje 2
        num_puntos = puntos_originales.size(2)
        global_feat_expanded = global_feat.repeat(1, 1, num_puntos) 
        # Ahora global_feat_expanded es (Batch, 1024, 8192)


        # CONCATENACIÓN (Global + Local)
        # ---------------------------------------------------------
        # Pegamos el contexto global con la geometría original
        combined_feat = torch.cat([puntos_originales, global_feat_expanded], dim=1)
        # combined_feat es (Batch, 1030, 8192)


        # DECODER (Puntuar cada punto)
        # ---------------------------------------------------------
        # Pasamos cada uno de los 8192 puntos por el MLP final
        scores_raw = self.decoder(combined_feat) 
        # Salida: (Batch, 1, 8192)

        # Función de activación Sigmoid: Mapea los valores crudos (logits) al rango [0, 1].
        # Esto es estrictamente necesario porque nuestro Ground Truth (generado previamente)
        # representa una probabilidad de viabilidad de agarre acotada entre 0 y 1. 
        # Sin esta función, la salida lineal del modelo desestabilizaría el cálculo del error (Loss).
        scores_prob = self.sigmoid(scores_raw)

        # Con `squeeze`quitamos todas las dimensiones de tamaño 1 (que sobra) para dejarlo limpio: (Batch, 8192)
        return scores_prob.squeeze(1)
    
