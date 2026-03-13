import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

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

# Map node IDs to indices
node_mapping = {node: i for i, node in enumerate(G.nodes())}

edges = []
for u, v in G.edges():
    edges.append([node_mapping[u], node_mapping[v]])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Node feature matrix
x = torch.tensor(df_encoded.values, dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

# Define GNN
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

model = GNN(x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()

    embeddings = model(data)

    src = embeddings[edge_index[0]]
    dst = embeddings[edge_index[1]]

    score = (src * dst).sum(dim=1)

    loss = F.binary_cross_entropy_with_logits(score, torch.ones_like(score))

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item()}")

# Final embeddings
emb = embeddings.detach().numpy()

sim_matrix = cosine_similarity(emb)

def recommend(item_index, top_k=3):
    scores = sim_matrix[item_index]
    similar_items = scores.argsort()[::-1][1:top_k+1]
    return similar_items

print("\nRecommended items for item 0:")
print(recommend(0))