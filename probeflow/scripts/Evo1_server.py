import sys
import os
import asyncio
import websockets
import numpy as np
import cv2
import json
import torch
from PIL import Image
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis
from safetensors.torch import load_file as safe_load_file


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1 import EVO1



class Normalizer:
    def __init__(self, stats_or_path):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        def pad_to_24(x):
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[0] < 24:
                pad = torch.zeros(24 - x.shape[0], dtype=torch.float32)
                x = torch.cat([x, pad], dim=0)
            elif x.shape[0] > 24:
                raise ValueError(f"Input length {x.shape[0]} exceeds expected 24")
            return x

        if len(stats) != 1:
            raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")

        robot_key = list(stats.keys())[0]
        robot_stats = stats[robot_key]

        self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
        self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
        self.action_min = pad_to_24(robot_stats["action"]["min"])
        self.action_max = pad_to_24(robot_stats["action"]["max"])

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state_min = self.state_min.to(state.device, dtype=state.dtype)
        state_max = self.state_max.to(state.device, dtype=state.dtype)
        return torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.action_min.to(action.device, dtype=action.dtype)
        action_max = self.action_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1)
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


def load_model_and_normalizer(ckpt_dir):
    config = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))

    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["num_inference_timesteps"] = 50

    model = EVO1(config).eval()
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    deepspeed_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

    if os.path.exists(safetensors_path):
        print(f"Loading from safetensors: {safetensors_path}")
        state_dict = safe_load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=True)
    elif os.path.exists(deepspeed_path):
        print(f"Loading from DeepSpeed checkpoint: {deepspeed_path}")
        checkpoint = torch.load(deepspeed_path, map_location="cpu")
        model.load_state_dict(checkpoint["module"], strict=True)
    else:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}. Expected model.safetensors or mp_rank_00_model_states.pt")
    model = model.to("cuda")

    normalizer = Normalizer(stats)
    return model, normalizer



def decode_image_from_list(img_list):
    img_array = np.array(img_list, dtype=np.uint8)
    img = cv2.resize(img_array, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    return transforms.ToTensor()(pil).to("cuda")



def infer_from_json_dict(data: dict, model, normalizer):
    device = "cuda"
    model_dtype = next(model.parameters()).dtype
    images = [decode_image_from_list(img) for img in data["image"]]
    state = torch.tensor(data["state"], dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[1] < 24:
        state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)
    norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor([data["action_mask"]],dtype=torch.int32, device=device)

    steps = data.get("steps", None)
    solver = data.get("solver", "rk")
    if solver == "rk":
        solver = "rk45"
    dt_probe = data.get("dt_probe", 0.5)

    with torch.no_grad() and torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action_raw, latency_breakdown, metadata = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask,
            steps=steps,
            solver=solver,
            dt_probe=dt_probe
        )

        action_raw = action_raw.reshape(1, -1, 24)
        action_denorm = normalizer.denormalize_action(action_raw[0])

        return {
            "action": action_denorm.cpu().numpy().tolist(),
            "latency_total": latency_breakdown["total"],
            "latency_vlm": latency_breakdown["vlm"],
            "latency_action": latency_breakdown["action"],
            "steps": metadata.get("steps", 0),
            "sim": metadata.get("sim", 0.0),
            "mag": metadata.get("mag", 0.0),
            "log_sqrt_var": metadata.get("log_sqrt_var", 0.0),
            "dt_probe": dt_probe,
        }


async def handle_request(websocket, model, normalizer):
    print("Client connected")
    try:
        async for message in websocket:

            json_data = json.loads(message)
            actions = infer_from_json_dict(json_data, model, normalizer)
            await websocket.send(json.dumps(actions))


    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evo1 Server")
    parser.add_argument("--port", type=int, default=9010, help="Port to run the server on")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to checkpoint directory")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    port = args.port

    print("Loading EVO_1 model...")
    model, normalizer = load_model_and_normalizer(ckpt_dir)

    async def main():
        print(f"EVO_1 server running at ws://0.0.0.0:{port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer),
            "0.0.0.0", port, max_size=100_000_000
        ):
            await asyncio.Future()

    asyncio.run(main())
