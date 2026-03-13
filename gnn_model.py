import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("data.csv")

# Encode categorical features
df_encoded = pd.get_dummies(df[["category", "color", "style"]])

# Create graph
G = nx.Graph()

for _, row in df.iterrows():
    G.add_node(row["id"])

# Create compatibility edges
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        same_color = df.loc[i, "color"] == df.loc[j, "color"]
        same_style = df.loc[i, "style"] == df.loc[j, "style"]

        if same_color or same_style:
            G.add_edge(df.loc[i, "id"], df.loc[j, "id"])

# Map node ids → indices
node_mapping = {node: i for i, node in enumerate(G.nodes())}

edges = []
for u, v in G.edges():
    edges.append([node_mapping[u], node_mapping[v]])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Node feature matrix (real encoded features)
x = torch.tensor(df_encoded.values, dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

# Define GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 8)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Initialize model
model = GNN(x.shape[1])

# Forward pass
embeddings = model(data)

print("Node embeddings:")
print(embeddings)

# Convert embeddings to numpy
emb = embeddings.detach().numpy()

# Similarity matrix
sim_matrix = cosine_similarity(emb)

def recommend(item_index, top_k=3):
    scores = sim_matrix[item_index]
    similar_items = scores.argsort()[::-1][1:top_k+1]
    return similar_items

print("\nRecommended items for item 0:")
print(recommend(0))