import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 

data = pd.read_csv("Mall_Customers.csv")
#print(data)

features = ["Annual Income (k$)", "Spending Score (1-100)"]

x=data[features]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

wcss = []
k_range = range(1,11)

for k in k_range:
    k_means = KMeans(n_clusters=k, init="k-means++", random_state=42)
    k_means.fit(x_scaled)
    wcss.append(k_means.inertia_)
    
#plt.figure(figsize=(8, 5))
#plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
#plt.title('Elbow Method for Optimal K')
#plt.xlabel('Number of Clusters (K)')
#plt.ylabel('WCSS (Inertia)')
#plt.xticks(k_range)
#plt.grid(True)
#plt.show ()
    
optimal_k = 5 
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)

data["Cluster_Labels"] = final_kmeans.fit_predict(x_scaled)

data.to_csv(r'C:\Users\MySQLAdmin\Documents\data_analysis\customers_clustered_output.csv', index=False)
print('Clustering Complet! Output saved to customers_clustered_output.csv')

centers = scaler.inverse_transform(final_kmeans.cluster_centers_)

print("\nCluster Centers")
print(pd.DataFrame(
    centers,
    columns=["Annual Income (k$)", "Spending Score (1-100)"]
))

plt.figure(figsize=(8,6))

colors = ["blue", "orange", "green", "red", "purple"]

for cluster in range(optimal_k):
    cluster_data = data[data["Cluster_Labels"] == cluster]

    plt.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        color=colors[cluster],
        s=70,
        label=f"Cluster {cluster+1}"
    )
    
centers = scaler.inverse_transform(final_kmeans.cluster_centers_)

plt.scatter(
    centers[:,0],
    centers[:,1],
    color="black",
    marker="o",
    s=50,
    label="Centroids"
    
)

for j, (x, y) in enumerate(centers):
    
    plt.text(
        x+2,
        y+2,
        f"C{j+1}\n({x:.1f}, {y:.1f})",
        fontsize=9,
        fontweight="bold",
        color="black",
        
    )    

plt.title("The Customers Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.legend(title="Customer's Groups")
plt.grid(True)

plt.show()

print("\nCustomers in Each Cluster")

for cluster in range(optimal_k):

    print(f"\n========== Cluster {cluster+1} ==========")

    customers = data[data["Cluster_Labels"] == cluster]

    print(customers[
        [
            "CustomerID",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ])
    
    print("\nNumber of Customers in Each Cluster")

cluster_counts = data["Cluster_Labels"].value_counts().sort_index()

for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster+1}: {count} customers")