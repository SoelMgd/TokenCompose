import json

with open("ade20k_obj_comp_5_1k.json", "r") as f:
    data = json.load(f)

# Garder les 100 premiers prompts
reduced_data = data[:5]

with open("ade20k_obj_comp_5_5.json", "w") as f:
    json.dump(reduced_data, f, indent=4)
