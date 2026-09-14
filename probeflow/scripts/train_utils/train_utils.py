import yaml
import logging
import swanlab
import matplotlib.pyplot as plt
from accelerate import Accelerator, DistributedDataParallelKwargs
import warnings
import matplotlib.pyplot as plt
import io
from PIL import Image
import torch
import torch.nn.functional as F
import os

accelerator = Accelerator()
SWANLAB_ENABLED = True

def load_yaml_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

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

def visualize_trajectory_3d(pred_action, gt_action, step, prompt):
    pred_xyz = pred_action[:, :3]
    gt_xyz = gt_action[:, :3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], 
            label='Ground Truth', color='green', linewidth=2, alpha=0.7)
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], color='green', marker='o', s=50)
    ax.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], color='green', marker='^', s=50)

    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], 
            label='Prediction', color='red', linewidth=2, alpha=0.9, linestyle='--')
    ax.scatter(pred_xyz[0, 0], pred_xyz[0, 1], pred_xyz[0, 2], color='red', marker='o', s=50)
    ax.scatter(pred_xyz[-1, 0], pred_xyz[-1, 1], pred_xyz[-1, 2], color='red', marker='^', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Step {step}: {prompt}\nGreen=GT, Red=Pred (Circle=Start, Triangle=End)")
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def setup_logging(log_dir: str) -> str:
    from datetime import datetime
    import logging, os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.log")
    if accelerator is None or accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_path}")
    return log_path

def init_swanlab(config: dict, accelerator: Accelerator):
    global SWANLAB_ENABLED

    if config.get("disable_wandb", False) or os.environ.get("SWANLAB_MODE", "").lower() == "disabled":
        SWANLAB_ENABLED = False
        if accelerator is None or accelerator.is_main_process:
            logging.info("SwanLab disabled by config/env; using local logs only.")
        return

    if accelerator is None or accelerator.is_main_process:
        try:
            swanlab.init(
                project=config.get("wandb_project"),
                name=config.get("run_name"),
                config=config
            )
        except Exception as exc:
            SWANLAB_ENABLED = False
            logging.warning(f"SwanLab init failed, falling back to local logs only: {exc}")

def log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator, mse=None, mae=None, images=None):
    current_epoch = step / len(dataloader)
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Estimated Epoch: {current_epoch:.2f}")
        logging.info(f"[Step {step}] Loss: {loss.item():.4f} | MSE: {mse if mse else 0:.4f} | MAE: {mae if mae else 0:.4f}")
        log_dict = {
            "step": step,
            "loss": loss.item(),
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "grad_norm/total": total_norm,
            "grad_norm/clipped": clipped_norm,
        }

        if mse is not None:
            log_dict["metrics/mse"] = mse
        if mae is not None:
            log_dict["metrics/mae"] = mae

        if SWANLAB_ENABLED and images is not None and isinstance(images, dict):
            for key, img in images.items():
                log_dict[key] = swanlab.Image(img, caption=key)
        if SWANLAB_ENABLED:
            swanlab.log(log_dict)

def get_with_warning(config: dict, key: str, default):
    if key in config:
        return config[key]
    else:
        warnings.warn(f"'{key}' not found in config, using default: {default!r}")
        return default

def evaluate_and_log_metrics(
    accelerator, 
    model, 
    step, 
    loss_mse, 
    pred_velocity_mask, 
    target_velocity, 
    action_mask, 
    batch, 
    prompts, 
    scheduler, 
    dataloader,
    config,
    total_norm, 
    clipped_norm,
    visualize=False 
):
    with torch.no_grad():
        loss_mae_raw = F.l1_loss(pred_velocity_mask, target_velocity, reduction='none')
        mae_score = (loss_mae_raw * action_mask).sum() / (action_mask.sum() + 1e-8)

        mae_val = mae_score.item()
        mse_val = loss_mse.item()

    vis_images = None

    if visualize and accelerator.is_main_process:
        vis_images = {}
        try:
            idx = 0
            prompt = prompts[idx]

            unwrapped_model = accelerator.unwrap_model(model)

            state_input = batch["states"][idx].unsqueeze(0).to(device=accelerator.device, dtype=torch.bfloat16) 
            image_mask_input = batch["image_masks"][idx].unsqueeze(0).to(device=accelerator.device)
            action_mask_input = batch["action_mask"][idx].unsqueeze(0).to(device=accelerator.device, dtype=torch.bfloat16)
            image_mask_input = batch["image_masks"][idx].to(device=accelerator.device) 
            images_input = batch["images"][idx]

            inference_steps = 10 

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated_action, _, _ = unwrapped_model.run_inference(
                    images=images_input,
                    image_mask=image_mask_input,
                    prompt=prompt,
                    state_input=state_input,
                    action_mask=action_mask_input,
                    steps=inference_steps
                )

            gt_action = batch["actions"][idx].detach().float().cpu() 

            if isinstance(generated_action, torch.Tensor):
                pred_action = generated_action.detach().squeeze(0).float().cpu()
            else:
                pred_action = torch.tensor(generated_action).squeeze(0).float().cpu()

            if pred_action.ndim == 1 and gt_action.ndim == 2:
                try:
                    pred_action = pred_action.view(-1, gt_action.shape[-1])
                except Exception as e:
                    logging.warning(f"Reshape failed: pred={pred_action.shape}, gt={gt_action.shape}, err={e}")

            traj_img = visualize_trajectory_3d(
                pred_action=pred_action, 
                gt_action=gt_action, 
                step=step, 
                prompt=f"{prompt} (Steps={inference_steps})"
            )
            vis_images = {"trajectory_comparison": traj_img}

        except Exception as e:
            logging.warning(f"Visualization failed at step {step}: {e}")
            import traceback
            traceback.print_exc()

    log_training_step(
        step, 
        loss_mse, 
        total_norm, 
        clipped_norm,
        scheduler, 
        dataloader, 
        accelerator,
        mse=mse_val,
        mae=mae_val,
        images=vis_images  
    )
