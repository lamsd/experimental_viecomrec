
import pandas as pd

# Load original CSVs
products = pd.read_csv("data_product.csv")
attrs = pd.read_csv("data_product_attribute.csv")
inter = pd.read_csv("data_reviews_purchase.csv")

# Build canonical items table with description field
prod_keep = ["product_id", "product_name", "brand", "origin", "type", "skin_kind", "shop_id"]
products_small = products[prod_keep].drop_duplicates("product_id")

attr_keep = ["product_id", "ingredient", "feature", "skin_type", "capacity", "design", "brand", "expiry", "origin"]
attrs_small = attrs[attr_keep].drop_duplicates("product_id")

items = products_small.merge(attrs_small, on="product_id", how="left", suffixes=("", "_attr"))
items["brand_final"] = items["brand_attr"].fillna(items["brand"])
items["origin_final"] = items["origin_attr"].fillna(items["origin"])

def build_desc(row):
    parts = [
        str(row.get("product_name", "")),
        f"brand: {row.get('brand_final','')}",
        f"origin: {row.get('origin_final','')}",
        f"type: {row.get('type','')}",
        f"skin_kind: {row.get('skin_kind','')}",
        f"ingredient: {row.get('ingredient','')}",
        f"feature: {row.get('feature','')}",
        f"skin_type: {row.get('skin_type','')}",
        f"capacity: {row.get('capacity','')}",
        f"design: {row.get('design','')}",
        f"expiry: {row.get('expiry','')}",
    ]
    return ". ".join([p for p in parts if p and p != "nan"]).strip()

items["description"] = items.apply(build_desc, axis=1)
items_final = items[["product_id", "product_name", "description"]].dropna(subset=["product_id"]).copy()
items_final.rename(columns={"product_id":"item_id", "product_name":"title"}, inplace=True)
items_final.to_csv("items_prepared.csv", index=False)

# Prepare interactions (user-item-timestamp)
inter = inter.rename(columns={"cmt_date":"timestamp", "product_id":"item_id"})
inter["timestamp"] = pd.to_datetime(inter["timestamp"], errors="coerce")
inter_small = inter[["user_id","item_id","timestamp"]].dropna(subset=["user_id","item_id"]).copy()

# Per-user split: train/val/test
df = inter_small.sort_values(["user_id","timestamp"]).copy()
df["pos"] = df.groupby("user_id").cumcount()
sizes = df.groupby("user_id")["item_id"].transform("size")
df["is_test"] = df["pos"] == (sizes - 1)
df["is_val"] = df["pos"] == (sizes - 2)
df["is_train"] = df["pos"] < (sizes - 2)

train_df = df[df["is_train"]][["user_id","item_id","timestamp"]].copy()
val_df   = df[df["is_val"]][["user_id","item_id","timestamp"]].copy().rename(columns={"item_id":"val_item"})
test_df  = df[df["is_test"]][["user_id","item_id","timestamp"]].copy().rename(columns={"item_id":"test_item"})

train_df.to_csv("interactions_train.csv", index=False)
val_df.to_csv("interactions_val.csv", index=False)
test_df.to_csv("interactions_test.csv", index=False)

print("Prepared items_prepared.csv, interactions_{train,val,test}.csv")
