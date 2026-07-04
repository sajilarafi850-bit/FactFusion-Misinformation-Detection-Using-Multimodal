import zipfile
import json

zip_path = "models/Text_Final_Model.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open("final_model/config.json") as f:
        config = json.load(f)
    with z.open("final_model/training_summary.json") as f:
        summary = json.load(f)

print("=== config.json ===")
print("id2label:", config.get("id2label"))
print("label2id:", config.get("label2id"))

print("\n=== training_summary.json ===")
print(json.dumps(summary, indent=2))