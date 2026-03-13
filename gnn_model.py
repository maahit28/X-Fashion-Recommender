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
df_encoded = pd.get_dummies(df[[
    "category",
    "color",
    "style",
    "season",
    "occasion",
    "gender",
    "material",
    "brand",
    "price_range"
]])

# Build graph
G = nx.Graph()

for _, row in df.iterrows():
    G.add_node(row["id"])

for i in range(len(df)):
    for j in range(i + 1, len(df)):

        same_color = df.loc[i, "color"] == df.loc[j, "color"]
        same_style = df.loc[i, "style"] == df.loc[j, "style"]
        same_season = df.loc[i, "season"] == df.loc[j, "season"]

        if same_color or same_style or same_season:
            G.add_edge(df.loc[i, "id"], df.loc[j, "id"])

# Map node ids
node_mapping = {node: i for i, node in enumerate(G.nodes())}

edges = []
for u, v in G.edges():
    edges.append([node_mapping[u], node_mapping[v]])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Node features
x = torch.tensor(df_encoded.values, dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

# Define GNN
class GNN(torch.nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 16)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        return x

model = GNN(x.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train GNN
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
        print("Epoch", epoch, "Loss:", loss.item())

# Generate embeddings
emb = embeddings.detach().numpy()

# Similarity matrix
sim_matrix = cosine_similarity(emb)

# Recommendation function
def recommend(item_index, top_k=3):

    scores = sim_matrix[item_index]

    similar_items = scores.argsort()[::-1][1:top_k+1]

    return similar_items

# Explanation function
def explain_recommendation(base_item, recommended_item):

    reasons = []

    if df.loc[base_item, "color"] == df.loc[recommended_item, "color"]:
        reasons.append("same color")

    if df.loc[base_item, "style"] == df.loc[recommended_item, "style"]:
        reasons.append("similar style")

    if df.loc[base_item, "season"] == df.loc[recommended_item, "season"]:
        reasons.append("same season")

    similarity_score = sim_matrix[base_item][recommended_item]

    return reasons, similarity_score


# Outfit generator
def generate_outfit(base_item):

    top_items = ["shirt", "tshirt", "hoodie", "sweater"]
    bottom_items = ["jeans", "trousers", "shorts"]
    outerwear_items = ["jacket", "coat", "blazer"]

    top = None
    bottom = None
    outerwear = None

    recs = recommend(base_item, top_k=15)

    for r in recs:

        category = df.loc[r, "category"]

        if category in top_items and top is None:
            top = r

        elif category in bottom_items and bottom is None:
            bottom = r

        elif category in outerwear_items and outerwear is None:
            outerwear = r

    return top, bottom, outerwear


# Test recommendation
base_item = 0

print("\nRecommendations for item 0:\n")

recs = recommend(base_item)

for r in recs:

    reasons, score = explain_recommendation(base_item, r)

    print("Recommended item:", df.loc[r, "category"])
    print("Color:", df.loc[r, "color"])
    print("Style:", df.loc[r, "style"])
    print("Similarity score:", round(score,3))
    print("Reason:", reasons)
    print("-----")


# Generate outfit
print("\nGenerated Outfit:\n")

top, bottom, outerwear = generate_outfit(base_item)

if top is not None:
    print("Top:", df.loc[top, "category"], "-", df.loc[top, "color"])

if bottom is not None:
    print("Bottom:", df.loc[bottom, "category"], "-", df.loc[bottom, "color"])

if outerwear is not None:
    print("Outerwear:", df.loc[outerwear, "category"], "-", df.loc[outerwear, "color"])