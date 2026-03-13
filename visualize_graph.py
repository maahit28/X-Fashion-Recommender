import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

G = nx.Graph()

for _, row in df.iterrows():
    G.add_node(row["id"], category=row["category"])

for i in range(len(df)):
    for j in range(i + 1, len(df)):

        if df.loc[i,"color"] == df.loc[j,"color"] or df.loc[i,"style"] == df.loc[j,"style"]:
            G.add_edge(df.loc[i,"id"], df.loc[j,"id"])

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

plt.figure(figsize=(10,8))

pos = nx.spring_layout(G)

nx.draw(
    G,
    pos,
    node_size=50,
    node_color="skyblue",
    edge_color="gray",
    with_labels=False
)

plt.title("Fashion Compatibility Graph")

plt.show()