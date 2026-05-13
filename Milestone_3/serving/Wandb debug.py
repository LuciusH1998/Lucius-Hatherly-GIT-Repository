import wandb, joblib, os

# Auto-login so you are not asked anything
wandb.login(key="50c6486ca894b323e5061c4513b477036ac87f8a", relogin=True)

api = wandb.Api()

for name in ["logreg_distance", "logreg_angle", "logreg_distance_angle"]:
    art = api.artifact(f"IFT6758-2025-B08/IFT6758-Milestone2/{name}:latest")
    d = art.download()
    for f in os.listdir(d):
        if f.endswith(".pkl"):
            m = joblib.load(os.path.join(d, f))
            print(name, getattr(m, "feature_names_in_", None))



