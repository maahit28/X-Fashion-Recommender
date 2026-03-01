import pandas as pd
import networkx as nx

# Load dataset
df = pd.read_csv("data.csv")

# Create graph
G = nx.Graph()

# Add nodes with attributes
for _, row in df.iterrows():
    G.add_node(
        row['id'],
        category=row['category'],
        color=row['color'],
        style=row['style']
    )

# Add edges based on compatibility
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        same_color = df.loc[i, 'color'] == df.loc[j, 'color']
        same_style = df.loc[i, 'style'] == df.loc[j, 'style']

        if same_color or same_style:
            G.add_edge(df.loc[i, 'id'], df.loc[j, 'id'])

# Print graph details
print("📌 Nodes:")
print(G.nodes(data=True))

print("\n🔗 Edges:")
print(G.edges())

# Recommendation function
def recommend(item_id):
    return list(G.neighbors(item_id))

# Test recommendation
print("\n🎯 Recommendations for item 1:")
print(recommend(1))