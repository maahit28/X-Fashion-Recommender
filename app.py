import streamlit as st
import pandas as pd
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

st.title("X-Fashion: Explainable AI Fashion Recommender")

# Load dataset
df = pd.read_csv("data.csv")

# Encode features
df_encoded = pd.get_dummies(df[["category","color","style"]])

# Build graph
G = nx.Graph()
for _, row in df.iterrows():
    G.add_node(row["id"])

for i in range(len(df)):
    for j in range(i+1,len(df)):
        if df.loc[i,"color"]==df.loc[j,"color"] or df.loc[i,"style"]==df.loc[j,"style"]:
            G.add_edge(df.loc[i,"id"],df.loc[j,"id"])

node_mapping = {node:i for i,node in enumerate(G.nodes())}

edges=[]
for u,v in G.edges():
    edges.append([node_mapping[u],node_mapping[v]])

edge_index=torch.tensor(edges,dtype=torch.long).t().contiguous()

x=torch.tensor(df_encoded.values,dtype=torch.float)

data=Data(x=x,edge_index=edge_index)

class GNN(torch.nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.conv1=GCNConv(num_features,16)
        self.conv2=GCNConv(16,8)

    def forward(self,data):
        x,edge_index=data.x,data.edge_index
        x=self.conv1(x,edge_index).relu()
        x=self.conv2(x,edge_index)
        return x

model=GNN(x.shape[1])
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    emb=model(data)
    src=emb[edge_index[0]]
    dst=emb[edge_index[1]]
    score=(src*dst).sum(dim=1)
    loss=F.binary_cross_entropy_with_logits(score,torch.ones_like(score))
    loss.backward()
    optimizer.step()

embeddings=emb.detach().numpy()
sim_matrix=cosine_similarity(embeddings)

def recommend(item_index,top_k=3):
    scores=sim_matrix[item_index]
    return scores.argsort()[::-1][1:top_k+1]

def explain(base,reco):
    reasons=[]
    if df.loc[base,"color"]==df.loc[reco,"color"]:
        reasons.append("same color")
    if df.loc[base,"style"]==df.loc[reco,"style"]:
        reasons.append("similar style")
    if df.loc[base,"category"]==df.loc[reco,"category"]:
        reasons.append("same category")
    return reasons

item=st.selectbox("Select clothing item",df.index)

if st.button("Recommend"):
    recs=recommend(item)
    for r in recs:
        st.write("### Recommended:",df.loc[r,"category"])
        st.write("Color:",df.loc[r,"color"])
        st.write("Style:",df.loc[r,"style"])
        for reason in explain(item,r):
    st.write("•", reason)
        st.write("---")