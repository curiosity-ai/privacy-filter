import json

# This would ideally load opf and dump RMSNorm, etc.
# For simplicity, we create a dummy dump.
dump = {
    "rmsnorm_input": [1.0, 2.0, 3.0, 4.0],
    "rmsnorm_weight": [1.0, 1.0, 1.0, 1.0],
    "rmsnorm_eps": 1e-5
}
with open("test_dump.json", "w") as f:
    json.dump(dump, f)
