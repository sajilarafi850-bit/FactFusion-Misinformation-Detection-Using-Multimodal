import zipfile
import json

zip_path = "models/Text_Final_Model.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    # Pehle dekhte hain zip ke andar files ka structure kya hai
    print("Files in zip:")
    for name in z.namelist():
        print(" -", name)

    # Ab tokenizer.json dhoondo aur load karo
    tokenizer_path = "final_model/tokenizer.json"
    with z.open(tokenizer_path) as f:
        data = json.load(f)

    print("\nModel type:", data["model"]["type"])
    print("File loaded successfully from zip!")