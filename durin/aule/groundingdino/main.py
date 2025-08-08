import torch
from models.GroundingDINO.groundingdino import GroundingDINO
from util.misc import clean_state_dict


def load_model(model_checkpoint_path):
    model = GroundingDINO()
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    print(f"Num unmatched: {len(load_res[0])}, {load_res[0]}")
    print("-------------")
    print(f"Num missing: {len(load_res[1])}, {load_res[1]}")

    _ = model.eval()
    return model


model = load_model("fangorn1.pth")
