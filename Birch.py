import numpy as np
import time
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 1. Geração de Dados Sintéticos (Dataset de Personalidade - Big Five)
# Gerando 10.000 usuários, 5 dimensões (OCEAN), valores normalizados entre 0 e 1
np.random.seed(42)
num_users = 10000
num_features = 5
print(f"Gerando dataset sintético de {num_users} usuários com {num_features} traços de personalidade (Big Five)...")
X_users = np.random.rand(num_users, num_features)

# 2. Implementação do algoritmo d-means (Simulação do GrouPlanner)
def d_means_clustering(data, threshold):
    """
    Agrupa iterativamente os usuários com base em um limiar (threshold).
    Se a distância de um usuário para o cluster mais próximo for menor que 
    o threshold, ele entra no grupo. Caso contrário, ele forma um novo grupo.
    """
    clusters = [] # Lista contendo os índices dos usuários agrupados
    centroids = [] # Lista contendo os centros geométricos dos grupos (vetor 5D)
    labels = np.zeros(len(data), dtype=int)
    
    for i, point in enumerate(data):
        if not centroids:
            clusters.append([i])
            centroids.append(point)
            labels[i] = 0
            continue
            
        # Calcula a distância euclidiana do novo usuário para todos os grupos existentes
        distances = np.linalg.norm(np.array(centroids) - point, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        # Verifica se atende ao limiar de similaridade
        if min_dist <= threshold:
            clusters[min_idx].append(i)
            labels[i] = min_idx
            # Atualiza o perfil médio do grupo
            centroids[min_idx] = np.mean(data[clusters[min_idx]], axis=0)
        else:
            # Novo usuário tem perfil muito diferente -> Cria novo grupo
            clusters.append([i])
            centroids.append(point)
            labels[i] = len(centroids) - 1
            
    return labels

print("\n--- INICIANDO TESTES PRÁTICOS ---")

# Parâmetros escolhidos para os testes no hiperespaço de 5 Dimensões
birch_threshold = 0.5
birch_branching = 50
dmeans_threshold = 0.5

# ==========================================
# TESTE 1: BIRCH ALGORITHM
# ==========================================
print("\nExecutando BIRCH...")
start_time = time.time()
# Sem limite de n_clusters para testarmos a árvore CF de forma pura
birch_model = Birch(threshold=birch_threshold, branching_factor=birch_branching, n_clusters=None)
labels_birch = birch_model.fit_predict(X_users)
time_birch = time.time() - start_time

n_clusters_birch = len(np.unique(labels_birch))

# Calculando métricas de qualidade
sil_birch = silhouette_score(X_users, labels_birch)
db_birch = davies_bouldin_score(X_users, labels_birch)

# ==========================================
# TESTE 2: D-MEANS (GrouPlanner)
# ==========================================
print("Executando d-means (Simulação GrouPlanner)...")
start_time = time.time()
labels_dmeans = d_means_clustering(X_users, dmeans_threshold)
time_dmeans = time.time() - start_time

n_clusters_dmeans = len(np.unique(labels_dmeans))

# Calculando métricas de qualidade
sil_dmeans = silhouette_score(X_users, labels_dmeans)
db_dmeans = davies_bouldin_score(X_users, labels_dmeans)

# ==========================================
# RESULTADOS
# ==========================================
print("\n" + "="*60)
print(" RESULTADOS DO COMPARATIVO (Persona Tour - PIBIT 2025/2026)")
print("="*60)
print(f"Dataset: {num_users} usuários (Vetor 5D Big Five)")
print("-" * 60)
print(f"{'Métrica':<20} | {'BIRCH':<15} | {'d-means':<15}")
print("-" * 60)
print(f"{'Clusters Gerados':<20} | {n_clusters_birch:<15} | {n_clusters_dmeans:<15}")
print(f"{'Tempo de Execução':<20} | {time_birch:.4f} s      | {time_dmeans:.4f} s")
print(f"{'Silhouette Score':<20} | {sil_birch:.4f}          | {sil_dmeans:.4f}")
print(f"{'Davies-Bouldin':<20} | {db_birch:.4f}          | {db_dmeans:.4f}")
print("="*60)