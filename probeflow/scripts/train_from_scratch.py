import sys
import os
import math
from torch import amp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import wandb
import swanlab
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint
_original_checkpoint = torch.utils.checkpoint.checkpoint
def _patched_checkpoint(*args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return _original_checkpoint(*args, **kwargs)
torch.utils.checkpoint.checkpoint = _patched_checkpoint

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from Evo1 import EVO1
from accelerate import Accelerator 
import logging
from datetime import datetime
import argparse
from accelerate import Accelerator, DistributedType
import json
import shutil
from torch.optim import AdamW
from scripts.train_utils.train_utils import(
    setup_logging,
    init_swanlab,
    get_with_warning,
    load_yaml_config,
    log_training_step, 
    evaluate_and_log_metrics,
)
import warnings
from safetensors.torch import load_file as safe_load_file
from model.action_head.adaflow import adaflow_loss

accelerator = Accelerator()

def inspect_named_submodules(module_dict: dict, verbose: bool = True):

    total_all, trainable_all = 0, 0
    logging.info("\n Parameter Inspection by Module:")
    logging.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logging.info(f"\n Module: {module_name}")
        logging.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logging.info(f"Trainable {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
            elif verbose:
                logging.info(f"Frozen {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
        logging.info("-" * 70)
        logging.info(f"Total     : {total / 1e6:.2f}M")
        logging.info(f"Trainable : {trainable / 1e6:.2f}M")
        logging.info(f"Frozen    : {(total - trainable) / 1e6:.2f}M")
        total_all += total
        trainable_all += trainable
    logging.info("=" * 70)
    logging.info(f"ALL TOTAL     : {total_all / 1e6:.2f}M")
    logging.info(f"ALL TRAINABLE : {trainable_all / 1e6:.2f}M")
    logging.info(f"ALL FROZEN    : {(total_all - trainable_all) / 1e6:.2f}M")
    logging.info("=" * 70)

def custom_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    images = [item["images"] for item in batch]
    states = torch.stack([item["state"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    action_mask = torch.stack([item["action_mask"] for item in batch], dim=0)
    image_masks = torch.stack([item["image_mask"] for item in batch], dim=0)
    state_mask = torch.stack([item["state_mask"] for item in batch], dim=0)
    embodiment_ids = torch.stack([item["embodiment_id"] for item in batch], dim=0)

    return {
        "prompts": prompts,
        "images": images,
        "states": states,
        "actions": actions,
        "action_mask": action_mask,
        "state_mask": state_mask,
        "image_masks": image_masks,
        "embodiment_ids": embodiment_ids
    }

def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    def lr_lambda(current_step):
        current_step += resume_step  
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda

def prepare_dataset(config: dict) -> torch.utils.data.Dataset:
    from dataset.read_data import TaskReader, compute_task_stats, EvoRealDataset
    from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset

    horizon = get_with_warning(config, "horizon", 16)
    task_dir = config.get("data_paths")
    prompt = config.get("prompt", "default task prompt")
    dataset_type = str(config.get("dataset_type", "auto")).lower()
    dataset_config_path = config.get("dataset_config_path")

    real_data_detected = False
    if task_dir and os.path.isdir(task_dir):
        real_data_detected = any(
            entry.startswith("episode_") and os.path.isdir(os.path.join(task_dir, entry))
            for entry in os.listdir(task_dir)
        )

    if dataset_type == "real" or real_data_detected:
        if dataset_type not in {"real", "auto"} and accelerator.is_main_process:
            logging.warning(
                "dataset_type=%s but data_paths looks like legacy real-robot episodes. Falling back to real dataset loader.",
                dataset_type,
            )

        task_reader = TaskReader(task_dir)
        norm_stats = compute_task_stats(task_reader)
        dataset = EvoRealDataset(
            task_reader=task_reader,
            norm_stats=norm_stats,
            prompt=prompt,
            horizon=horizon,
            max_state_dim=config.get("state_dim", 7),
            max_action_dim=config.get("action_dim", 7)
        )
        config["num_categories"] = 1
        if accelerator is None or accelerator.is_main_process:
            logging.info(f"Loaded {len(dataset)} samples from {task_dir} (real)")
            logging.info("num_categories auto-set to 1")
        return dataset

    if dataset_type not in {"auto", "lerobot"}:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    if not dataset_config_path:
        raise ValueError("LeRobot dataset requires --dataset_config_path")

    if not os.path.isabs(dataset_config_path):
        dataset_config_path = os.path.join(os.path.dirname(__file__), "..", dataset_config_path)
    dataset_cfg = load_yaml_config(dataset_config_path)

    dataset = LeRobotDataset(
        config=dataset_cfg,
        image_size=config.get("image_size", 448),
        action_horizon=horizon,
        binarize_gripper=config.get("binarize_gripper", False),
        use_augmentation=config.get("use_augmentation", False),
    )
    config["num_categories"] = max(1, len(getattr(dataset, "arm_to_embodiment_id", {})))
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Loaded {len(dataset)} samples from {dataset_config_path} (lerobot)")
        logging.info(f"num_categories auto-set to {config['num_categories']}")
    return dataset

def prepare_dataloader(dataset, config: dict) -> DataLoader:
    batch_size = get_with_warning(config, "batch_size", 8)
    num_workers = get_with_warning(config, "num_workers", 8)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Initialized dataloader with batch size {batch_size}")
    return dataloader

def check_numerical_stability(step: int, **named_tensors) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            logging.info(f"[Step {step}] Non-finite detected in {name}")
            return False
    return True

def save_checkpoint(save_dir, step, model_engine, loss, accelerator, config=None, norm_stats=None):
    tag = f"step_{step}"
    checkpoint_dir = os.path.join(save_dir, tag)

    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} exists. Removing before overwrite.")
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()

    if hasattr(model_engine, "save_checkpoint"):
        client_state = {
            "step": step,
            "best_loss": loss if isinstance(loss, float) else loss.item(),
            "config": config,
        } if accelerator.is_main_process else {}
        model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
    else:
        accelerator.save_state(checkpoint_dir)

    if accelerator.is_main_process:
        if config is not None:
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        if norm_stats is not None:
            norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.json")
            with open(norm_stats_path, "w") as f:
                json.dump(norm_stats, f, indent=2)

        checkpoint_meta_path = os.path.join(checkpoint_dir, "checkpoint.json")
        checkpoint_meta = {
            "type": "ds_model" if hasattr(model_engine, "save_checkpoint") else "ddp_model",
            "version": 0.0,
            "checkpoints": "mp_rank_00_model_states.pt"
        }
        with open(checkpoint_meta_path, "w") as f:
            json.dump(checkpoint_meta, f, indent=2)
        logging.info(f"[Rank {accelerator.process_index}] Saved checkpoint to {checkpoint_dir}")

def load_checkpoint_with_deepspeed(model_engine, load_dir, accelerator, tag="step_best", load_optimizer_states=True, resume_pretrain=False):
    ckpt_path = os.path.join(load_dir, tag)

    is_ds_checkpoint = os.path.exists(os.path.join(ckpt_path, "mp_rank_00_model_states.pt"))

    if hasattr(model_engine, "load_checkpoint") and is_ds_checkpoint:
        try:
            load_path, client_state = model_engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=True,
                load_optimizer_states=load_optimizer_states and not resume_pretrain,
                load_lr_scheduler_states=load_optimizer_states and not resume_pretrain
            )
            if accelerator.is_main_process:
                logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag}")
            return client_state.get("step", 0), client_state
        except Exception as e:
            if accelerator.is_main_process:
                logging.warning(f"DeepSpeed native load failed: {e}. Falling back to manual load...")


    if accelerator.is_main_process:
        logging.info(f"Loading checkpoint manually (weights only) from {ckpt_path} ...")

    model_path = os.path.join(ckpt_path, "model.safetensors")
    if not os.path.exists(model_path):
        model_path = os.path.join(ckpt_path, "pytorch_model.bin")
    if not os.path.exists(model_path) and is_ds_checkpoint:
        model_path = os.path.join(ckpt_path, "mp_rank_00_model_states.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model weights at {ckpt_path}. (Expected model.safetensors, pytorch_model.bin or mp_rank_*.pt)")

    if model_path.endswith(".safetensors"):
        state_dict = safe_load_file(model_path)
    else:
        loaded_obj = torch.load(model_path, map_location="cpu")
        state_dict = loaded_obj.get("module", loaded_obj) if isinstance(loaded_obj, dict) else loaded_obj

    unwrapped_model = accelerator.unwrap_model(model_engine)

    missing, unexpected = unwrapped_model.load_state_dict(state_dict, strict=False)

    if accelerator.is_main_process:
        logging.info(f"Manual weights loaded successfully.")
        logging.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing:
            logging.info(f"Missing sample: {missing[:8]}")
        if unexpected:
            logging.info(f"Unexpected sample: {unexpected[:8]}")
        if resume_pretrain:
            logging.info("Resume Pretrain Mode: Optimizer states were ignored. Training starts from step 0.")

    return 0, {}

def get_and_clip_grad_norm(accelerator, model, loss, max_norm: float = 1.0):

    if hasattr(accelerator, "get_global_grad_norm") and hasattr(accelerator, "clip_grad_norm_"):

        total_norm = accelerator.get_global_grad_norm()
        accelerator.clip_grad_norm_(model.parameters(), max_norm)
        clipped_norm = accelerator.get_global_grad_norm()
    else:

        grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(grad_norms) == 0:
            total_norm = torch.tensor(0.0, device=loss.device)
        else:
            total_norm = torch.norm(torch.stack(grad_norms), 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        clipped_grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(clipped_grad_norms) == 0:
            clipped_norm = torch.tensor(0.0, device=loss.device)
        else:
            clipped_norm = torch.norm(torch.stack(clipped_grad_norms), 2)

    return total_norm, clipped_norm

def build_param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = n.endswith("bias") or ".bias" in n
        is_norm = (p.dim() == 1) or ("norm" in n.lower())
        (no_decay if is_bias or is_norm else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]

def train(config):

    save_dir = get_with_warning(config, "save_dir", "checkpoints")
    log_path = setup_logging(save_dir)

    init_swanlab(config, accelerator)

    if get_with_warning(config, "debug", False):
        torch.autograd.set_detect_anomaly(True)

    use_adaflow = get_with_warning(config, "use_adaflow", False)
    adaflow_train_base = get_with_warning(config, "adaflow_train_base", False)

    resume = get_with_warning(config, "resume", False)
    resume_path = get_with_warning(config, "resume_path", None)
    resume_pretrain = get_with_warning(config, "resume_pretrain", False)

    if resume != bool(resume_path):
        raise ValueError("Inconsistent resume configuration: --resume and --resume_path must be set together.")

    if use_adaflow and not adaflow_train_base and not resume:
        raise ValueError(
            "AdaFlow variance-only training requires a pretrained base checkpoint. "
            "Use --resume --resume_path ... or enable --adaflow_train_base for joint training."
        )

    dataset = prepare_dataset(config)

    dataloader = prepare_dataloader(dataset, config)

    model = EVO1(config)
    model.train()
    model.set_finetune_flags()
    if accelerator.is_main_process:
        logging.info(
            "AdaFlow mode: use_adaflow=%s, adaflow_train_base=%s, num_categories=%s",
            use_adaflow,
            adaflow_train_base,
            config.get("num_categories", 1),
        )

    lr = get_with_warning(config, "lr", 1e-5)
    wd = get_with_warning(config, "weight_decay", 1e-5)
    optimizer = AdamW(build_param_groups(model, wd), lr=lr)
    if accelerator.is_main_process:
        logging.info(f"Optimizer=AdamW, lr={lr}, weight_decay={wd}")


    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model  

    if accelerator.is_main_process:
        logging.info("Initialized with Accelerate")


    max_steps = get_with_warning(config, "max_steps", 1000)
    warmup_steps = get_with_warning(config, "warmup_steps", 300)

    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_checkpoint.pt")
    best_loss = float("inf")

    log_interval = get_with_warning(config, "log_interval", 100)
    vis_interval = get_with_warning(config, "vis_interval", 100)
    ckpt_interval = get_with_warning(config, "ckpt_interval", 1000)
    max_norm = get_with_warning(config, "grad_clip_norm", 1.0)

    if resume:
        resume_path = resume_path.rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)

        step, client_state = load_checkpoint_with_deepspeed(
            model_engine,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            load_optimizer_states=True,  
            resume_pretrain=resume_pretrain
        )
        best_loss = client_state.get("best_loss", float("inf"))
        if accelerator.is_main_process:
            logging.info(f"Resuming from {resume_dir}/{resume_tag}, step {step}")
    else:
        step = 0
        if accelerator.is_main_process:
            logging.info("Starting fresh training")

    if resume_pretrain:
        step = 0
        logging.info("Resuming pretraining from scratch, resetting step to 0")

    scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps, resume_step=step))


    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        inspect_named_submodules({
            "vision_model": unwrapped_model.embedder.model.vision_model,
            "language_model": unwrapped_model.embedder.model.language_model,
            "action_head": unwrapped_model.action_head
        })

    while step < max_steps:
        for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_main_process):
            if step >= max_steps:
                break
            prompts = batch["prompts"]
            images_batch = batch["images"]
            image_masks = batch["image_masks"]
            states = batch["states"].to(dtype=torch.bfloat16)
            actions_gt = batch["actions"].to(dtype=torch.bfloat16)
            action_mask = batch["action_mask"]
            state_mask = batch["state_mask"]
            embodiment_ids = batch["embodiment_ids"]
            fused_tokens_list = []
            unwrapped_model = accelerator.unwrap_model(model)

            for prompt, images, image_mask in zip(prompts, images_batch, image_masks):
                fused = unwrapped_model.get_vl_embeddings(images=images, image_mask=image_mask, prompt=prompt, return_cls_only=False)
                fused_tokens_list.append(fused.to(dtype=torch.bfloat16))

            fused_tokens = torch.cat(fused_tokens_list, dim=0)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_outputs = model(
                    fused_tokens,
                    state=states,
                    actions_gt=actions_gt,
                    action_mask=action_mask,
                    embodiment_ids=embodiment_ids,
                )
                if use_adaflow:
                    pred_velocity, noise, log_sqrt_var = model_outputs
                else:
                    pred_velocity, noise = model_outputs

            target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)

            assert pred_velocity.shape == target_velocity.shape

            if action_mask.sum() == 0:
                raise ValueError(f"[Step {step}] action_mask.sum() is 0! All actions are masked. "
                            f"This indicates a problem with the data or mask generation. "
                            f"action_mask shape: {action_mask.shape}, "
                            f"action_mask: {action_mask}")


            action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
            pred_velocity_mask = pred_velocity * action_mask
            target_velocity_mask = target_velocity * action_mask

            if use_adaflow:
                error_target = target_velocity.detach() if not adaflow_train_base else target_velocity
                error_pred = pred_velocity.detach() if not adaflow_train_base else pred_velocity
                loss = adaflow_loss(
                    pred_velocity=error_pred,
                    target_velocity=error_target,
                    log_sqrt_var=log_sqrt_var,
                    action_mask=action_mask,
                    reduction="mean",
                )
            else:
                masked_sq_error = F.mse_loss(pred_velocity, target_velocity, reduction="none") * action_mask
                loss = masked_sq_error.sum() / (action_mask.sum() + 1e-8)

            if not check_numerical_stability(
                step,
                states=states,
                actions_gt=actions_gt,
                fused_tokens=fused_tokens,
                pred_velocity=pred_velocity,
                target_velocity=target_velocity,
                loss=loss
            ):
                continue

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)

            total_norm, clipped_norm = get_and_clip_grad_norm(accelerator, model, loss, max_norm)

            optimizer.step()
            scheduler.step()

            if step % log_interval == 0:
                do_visualize = (step % vis_interval == 0)
                evaluate_and_log_metrics(
                    accelerator=accelerator,
                    model=model,
                    step=step,
                    loss_mse=loss,              
                    pred_velocity_mask=pred_velocity_mask, 
                    target_velocity=target_velocity_mask,      
                    action_mask=action_mask,
                    batch=batch,     
                    prompts=prompts,
                    scheduler=scheduler,
                    dataloader=dataloader,
                    config=config,
                    total_norm=total_norm,
                    clipped_norm=clipped_norm,
                    visualize=do_visualize,  
                )

            loss_value = loss.item()
            if accelerator.is_main_process:
                is_best = loss_value < best_loss
                if is_best:
                    best_loss = loss_value
                is_best_tensor = torch.tensor(int(is_best), device=accelerator.device)
            else:
                is_best_tensor = torch.tensor(0, device=accelerator.device)

            if accelerator.distributed_type != DistributedType.NO:
                torch.distributed.broadcast(is_best_tensor, src=0)

            if is_best_tensor.item() == 1 and step > 1000:
                accelerator.print("start to save best checkpoint")
                save_checkpoint(
                    save_dir,
                    step="best",
                    model_engine=model_engine,
                    loss=loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=dataset.arm2stats_dict 
                )
                accelerator.print("end to save best checkpoint")
                if accelerator.is_main_process:
                    logging.info(f"Saved best checkpoint at step {step} with loss {loss_value:.6f}")

            step += 1

            if step % ckpt_interval == 0 and step > 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
                save_checkpoint(save_dir, step=step, model_engine=model_engine, loss=loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict)

    save_checkpoint(save_dir, step="final", model_engine=model_engine, loss=loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict)
    logging.info(f"Final model saved to step_final/")
    logging.info(f"Best checkpoint saved to step_best/ with loss {best_loss:.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Evo-1")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--vlm_name", type=str, default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--action_head", type=str, default="flowmatching", choices=["flowmatching"])
    parser.add_argument("--return_cls_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")

    parser.add_argument("--dataset_type", type=str, default="auto", choices=["auto", "real", "lerobot"])
    parser.add_argument("--data_paths", type=str, required=False)
    parser.add_argument("--dataset_config_path", type=str, required=False)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--binarize_gripper", action="store_true", default=False, help="Whether to binarize gripper state/action (default: False).")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation on images")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)


    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_pretrain", action="store_true")


    parser.add_argument("--finetune_vlm", action="store_true")
    parser.add_argument("--finetune_action_head", action="store_true")
    parser.add_argument("--use_adaflow", action="store_true")
    parser.add_argument("--adaflow_train_base", action="store_true")
    parser.add_argument("--adaflow_eta", type=float, default=0.5)
    parser.add_argument("--adaflow_min_steps", type=int, default=2)
    parser.add_argument("--adaflow_max_steps", type=int, default=50)

    parser.add_argument("--per_action_dim", type=int, default=7)
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_inference_timesteps", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="pick and place")
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    config = vars(args)

    try:
        train(config)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        sys.exit(0)
