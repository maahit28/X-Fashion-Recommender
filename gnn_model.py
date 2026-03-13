import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import random

# Load dataset
df = pd.read_csv("data.csv")

# Encode categorical features
df_encoded = pd.get_dummies(df[[
"category","color","pattern","style","fit","season",
"occasion","body_type","skin_tone","comfort_level",
"material","brand","price_range"
]])

# -------- GRAPH CONSTRUCTION (FAST VERSION) --------

G = nx.Graph()

for _, row in df.iterrows():
    G.add_node(row["id"])

# connect items with similar attributes using grouping
for col in ["color","style","season"]:
    
    groups = df.groupby(col)

    for _, group in groups:
        
        ids = group["id"].tolist()

        for i in range(len(ids)-1):
            G.add_edge(ids[i], ids[i+1])

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# map nodes
node_mapping = {node:i for i,node in enumerate(G.nodes())}

edges = []
for u,v in G.edges():
    edges.append([node_mapping[u],node_mapping[v]])

edge_index = torch.tensor(edges,dtype=torch.long).t().contiguous()

# features
x = torch.tensor(df_encoded.values,dtype=torch.float)

data = Data(x=x,edge_index=edge_index)

# -------- GNN MODEL --------

class GNN(torch.nn.Module):

    def __init__(self,num_features):
        super().__init__()

        self.conv1 = GCNConv(num_features,32)
        self.conv2 = GCNConv(32,16)

    def forward(self,data):

        x,edge_index = data.x,data.edge_index

        x = self.conv1(x,edge_index).relu()
        x = self.conv2(x,edge_index)

        return x

model = GNN(x.shape[1])
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

# -------- TRAIN --------

for epoch in range(50):

    optimizer.zero_grad()

    embeddings = model(data)

    src = embeddings[edge_index[0]]
    dst = embeddings[edge_index[1]]

    score = (src*dst).sum(dim=1)

    loss = F.binary_cross_entropy_with_logits(score,torch.ones_like(score))

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch",epoch,"Loss:",loss.item())

# -------- EMBEDDINGS --------

emb = embeddings.detach().numpy()
sim_matrix = cosine_similarity(emb)

# -------- RECOMMEND --------

def recommend(item_index,top_k=20):

    scores = sim_matrix[item_index]

    return scores.argsort()[::-1][1:top_k+1]


# -------- OUTFIT GENERATOR --------

tops = [
"tshirt","shirt","blouse","crop_top","tank_top","camisole",
"tube_top","halter_top","wrap_top","peplum_top","hoodie",
"sweater","cardigan","turtleneck","bodysuit"
]

bottoms = [
"jeans","skinny_jeans","wide_leg_jeans","mom_jeans",
"baggy_jeans","trousers","cargo_pants","leggings",
"palazzo","culottes","skirt","mini_skirt","midi_skirt",
"maxi_skirt","skort","shorts","denim_shorts","bike_shorts"
]

outerwear = [
"denim_jacket","leather_jacket","bomber_jacket",
"trench_coat","overcoat","parka","blazer","cape","puffer_jacket"
]

shoes = [
"sneakers","running_shoes","high_heels","stilettos",
"boots","ankle_boots","knee_boots","loafers",
"sandals","platforms","wedges","flats"
]

accessories = [
"necklace","choker","earrings","bracelet",
"watch","belt","bag","tote_bag",
"crossbody_bag","clutch","scarf",
"sunglasses","hairband","ring"
]


def generate_outfit(base_item):

    recs = recommend(base_item,top_k=50)

    outfit = {}

    for r in recs:

        cat = df.loc[r,"category"]

        if cat in tops and "top" not in outfit:
            outfit["top"] = r

        elif cat in bottoms and "bottom" not in outfit:
            outfit["bottom"] = r

        elif cat in outerwear and "outerwear" not in outfit:
            outfit["outerwear"] = r

        elif cat in shoes and "shoes" not in outfit:
            outfit["shoes"] = r

        elif cat in accessories and "accessory" not in outfit:
            outfit["accessory"] = r

        if len(outfit) == 5:
            break

    return outfit


# -------- TEST --------

base_item = random.randint(0,len(df)-1)

print("\nBase item:",df.loc[base_item,"category"],df.loc[base_item,"color"])

outfit = generate_outfit(base_item)

print("\nGenerated Outfit:\n")

for key,val in outfit.items():

    print(
        key.capitalize(),
        ":",
        df.loc[val,"category"],
        "-",
        df.loc[val,"color"]
    )