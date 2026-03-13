import pandas as pd
import random

categories = [
    "shirt","tshirt","hoodie","jacket","jeans",
    "trousers","blazer","sweater","coat","shorts"
]

colors = [
    "black","white","blue","grey","brown",
    "green","red","beige","navy","yellow"
]

styles = [
    "casual","formal","streetwear","sport","business"
]

seasons = [
    "summer","winter","spring","autumn"
]

occasions = [
    "daily","office","party","travel","gym"
]

genders = [
    "male","female","unisex"
]

materials = [
    "cotton","denim","wool","linen","polyester"
]

brands = [
    "nike","adidas","zara","hm","uniqlo",
    "puma","levi","gucci","prada","gap"
]

price_ranges = [
    "low","medium","high","premium"
]

data = []

for i in range(200):

    row = {
        "id": i,
        "category": random.choice(categories),
        "color": random.choice(colors),
        "style": random.choice(styles),
        "season": random.choice(seasons),
        "occasion": random.choice(occasions),
        "gender": random.choice(genders),
        "material": random.choice(materials),
        "brand": random.choice(brands),
        "price_range": random.choice(price_ranges)
    }

    data.append(row)

df = pd.DataFrame(data)

df.to_csv("data.csv", index=False)

print("Dataset with 200 rows generated successfully!")