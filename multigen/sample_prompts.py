import json
#"ade20k_obj_comp_5_1k.json"
with open("coco_obj_comp_5_1k.json", "r") as f:
    data = json.load(f)

# Garder les 100 premiers prompts
reduced_data = data[:50]

with open("coco_obj_comp_5_50.json", "w") as f:
    json.dump(reduced_data, f, indent=4)
