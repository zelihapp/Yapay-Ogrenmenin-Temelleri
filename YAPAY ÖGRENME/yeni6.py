import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx

# Excel dosyalarını yükle
file_path = "Corrected_Distance_Table_Reference.xlsx"
allocation_file_path = "Aid_Requests.xlsx"

# Mesafe ve talep tablolarını yükle
distances_df = pd.read_excel(file_path)
allocation_df = pd.read_excel(allocation_file_path)

# Kümeleme için ihtiyaç noktalarını al (Depolar hariç)
need_points_data = distances_df[~distances_df['Unnamed: 0'].str.contains('Depo_')].iloc[:, 1:].values
need_points_names = distances_df[~distances_df['Unnamed: 0'].str.contains('Depo_')]['Unnamed: 0'].values

# Küme sayısını belirle
num_clusters = 4

# KMeans algoritması ile kümeleme
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(need_points_data)

# Küme bilgilerini dataframe'e ekle
clusters_df = pd.DataFrame({
    'Point': need_points_names,
    'Cluster': cluster_labels
})

# Küme bilgilerini ihtiyaç noktalarına ekle
need_points = distances_df[~distances_df['Unnamed: 0'].str.contains('Depo_')].copy()
need_points = need_points.merge(clusters_df, left_on='Unnamed: 0', right_on='Point')

# Depoları belirle
depot_columns = [col for col in distances_df.columns if "Depo_" in col]

# Depo sütunlarının doğruluğunu kontrol et
if len(depot_columns) == 0:
    raise ValueError("Depo sütunları bulunamadı. Mesafe tablosunu kontrol edin.")

def calculate_dijkstra_route_and_distance(graph, depot_name, relevant_points):
    subgraph = graph.subgraph([depot_name] + relevant_points)
    shortest_paths = nx.single_source_dijkstra_path_length(subgraph, depot_name, weight="weight")
    route = [depot_name]
    visited = set(route)
    current_node = depot_name
    total_distance = 0

    while len(visited) < len(relevant_points) + 1:
        next_node = min(
            (node for node in relevant_points if node not in visited),
            key=lambda node: shortest_paths[node]
        )
        route.append(next_node)
        visited.add(next_node)
        total_distance += graph[current_node][next_node]['weight']
        current_node = next_node

    route.append(depot_name)
    total_distance += graph[current_node][depot_name]['weight']
    return route, total_distance

# Grafiği oluştur ve mesafe tablosunu ekle
graph = nx.DiGraph()
nodes = list(distances_df['Unnamed: 0'])
graph.add_nodes_from(nodes)

for i, source in enumerate(nodes):
    for j, target in enumerate(nodes):
        if i != j:
            distance = distances_df.iloc[i, j + 1]
            graph.add_edge(source, target, weight=distance)

# Her küme için rota hesaplama ve drone yönetimi


def manage_drone_for_route(route, allocation_df):
    drone_capacity = 30
    total_demand = 0
    drones_used = 0

    for point in route:
        if point.startswith('Depo'):
            continue

        # İsimlendirmeyi dönüştür
        point = point.replace('Point_', 'Ihtiyac_Noktasi_') if point.startswith('Point_') else point

        demand = allocation_df.loc[allocation_df['need_point'] == point, 'Medical_Supplies'].sum() + \
                 allocation_df.loc[allocation_df['need_point'] == point, 'Food_Supplies'].sum()

        print(f"Nokta: {point}, Talep: {demand}")
        total_demand += demand

    while total_demand > 0:
        drones_used += 1
        if total_demand <= drone_capacity:
            print(f"{drones_used}. Drone yola çıktı ve toplam {total_demand} birim ihtiyaç karşıladı.")
            break
        else:
            print(f"{drones_used}. Drone yola çıktı ve maksimum {drone_capacity} birim taşıdı.")
            total_demand -= drone_capacity

    return drones_used

# Küme bazında işlemleri uygula
routes = {}
drone_counts = {}
clusters = need_points['Cluster'].unique()

# Depoları liste olarak al ve doğrula
depot_list = distances_df.loc[distances_df['Unnamed: 0'].str.contains("Depo_"), 'Unnamed: 0'].tolist()
if not depot_list:
    raise ValueError("Hiçbir depo bulunamadı. Mesafe tablosunu kontrol edin.")

for cluster in clusters:
    cluster_points = need_points[need_points['Cluster'] == cluster]['Unnamed: 0'].tolist()
    depot = depot_list[cluster % len(depot_list)]
    print(f"\nKüme {cluster} için işlem başlıyor. İlgili depo: {depot}")
    print(f"Küme {cluster} içindeki noktalar: {cluster_points}")
    route, total_distance = calculate_dijkstra_route_and_distance(graph, depot, cluster_points)
    routes[cluster] = (route, total_distance)
    print(f"Küme {cluster} için hesaplanan rota: {route}")
    print(f"Küme {cluster} için toplam mesafe: {total_distance:.2f} km")
    drones_used = manage_drone_for_route(route, allocation_df)
    drone_counts[cluster] = drones_used
    print(f"Küme {cluster} için kullanılan drone sayısı: {drones_used}")

# Toplam drone sayısını yazdır
print(f"\nToplam kullanılan drone sayısı: {sum(drone_counts.values())}")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# PCA ile 2 boyuta indirgeme
pca = PCA(n_components=2)
points_2d = pca.fit_transform(need_points_data)
centroids_2d = pca.transform(kmeans.cluster_centers_)  # Küme merkezlerini dönüştürme

# İlk grafik: PCA tabanlı görselleştirme
plt.figure(figsize=(10, 7))

# Kümeleri renklendirme
colors = ['purple', 'blue', 'green', 'yellow']  # Kümeler için renkler
for cluster in range(num_clusters):
    cluster_points = points_2d[cluster_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[cluster], label=f"Küme {cluster}")

# Küme merkezlerini ekleme
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', s=200, marker='X', label="Küme Merkezleri")

# Grafik detayları
plt.title("Kümelerin ve Merkezlerin Görselleştirilmesi (2D PCA ile)", fontsize=14)
plt.xlabel("PCA Bileşeni 1", fontsize=12)
plt.ylabel("PCA Bileşeni 2", fontsize=12)
plt.legend(title="Kümeler", fontsize=10, loc="best")
plt.grid(True)
plt.show()

# İkinci grafik: Nokta indeksi tabanlı kümeleme görselleştirme
plt.figure(figsize=(10, 7))
plt.scatter(np.arange(len(cluster_labels)), cluster_labels, c=cluster_labels, cmap='viridis', s=50)
plt.colorbar(label="Küme")
plt.title("K-Means Kümeleme Sonuçları", fontsize=14)
plt.xlabel("Nokta İndeksi", fontsize=12)
plt.ylabel("Küme Etiketi", fontsize=12)
plt.show()


