import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from model.action_head.adaflow import AdaFlowVarianceHead, adaflow_sample

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int):
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :]

    def _extend_pe(self, new_max_len):
        old_max_len, dim = self.pe.size(1), self.pe.size(2)
        if new_max_len <= old_max_len:
            return
        extra_positions = torch.arange(old_max_len, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        extra_pe = torch.zeros(new_max_len - old_max_len, dim)
        extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
        extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
        extra_pe = extra_pe.unsqueeze(0)
        new_pe = torch.cat([self.pe, extra_pe.to(self.pe.device)], dim=1)
        self.pe = new_pe

class CategorySpecificLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1):
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.weight = nn.Parameter(torch.randn(num_categories, in_dim, out_dim))
            self.bias = nn.Parameter(torch.randn(num_categories, out_dim))

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):

        if self.num_categories <= 1:
            return self.linear(x)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]) 
        if category_id.dim() == 0:

            cid = category_id.item()
            out = x_flat @ self.weight[cid] + self.bias[cid]
        else:

            category_id = category_id.view(-1)  
            weight_selected = self.weight[category_id]        
            bias_selected = self.bias[category_id]        
            out = torch.bmm(x_flat.unsqueeze(1), weight_selected).squeeze(1) + bias_selected
        out_shape = orig_shape[:-1] + (out.shape[-1],)
        return out.view(out_shape)

class CategorySpecificMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_categories: int = 1):
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        out = self.activation(self.fc1(x, category_id))
        out = self.fc2(out, category_id)
        return out

class MultiEmbodimentActionEncoder(nn.Module):

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories

        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)

        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):

        B, H, D = action_seq.shape
        assert H == self.horizon, "Action sequence length must match horizon"

        x = action_seq.reshape(B * H, D) 

        if category_id.dim() == 0:

            cat_ids = category_id.repeat(H * B)
        else:
            cat_ids = category_id.unsqueeze(1).repeat(1, H).reshape(B * H)
        out = self.activation(self.W1(x, cat_ids))            

        pos_enc = self.pos_encoding(H).to(out.device)       
        pos_enc = pos_enc.repeat(B, 1, 1).reshape(B * H, -1) 
        out = out + pos_enc
        out = self.activation(self.W2(out, cat_ids))         
        out = self.W3(out, cat_ids)                        
        out = out.view(B, H, self.embed_dim)
        return out

class BasicTransformerBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, action_tokens: torch.Tensor, context_tokens: torch.Tensor, time_emb: torch.Tensor):

        x = self.norm1(action_tokens)
        attn_out, _ = self.attn(x, context_tokens, context_tokens)

        x = action_tokens + attn_out

        x2 = self.norm2(x)

        if time_emb is not None:
            x2 = x2 + time_emb.unsqueeze(1)
        ff_out = self.ff(x2)
        x = x + ff_out
        return x

class FlowmatchingActionHead(nn.Module):

    def __init__(self, config=None,
                 embed_dim: int = 896, 
                 hidden_dim: int = 1024,
                 action_dim: int = 16*7,
                 horizon: int = 16,
                 per_action_dim: int = 7,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.0,
                 num_inference_timesteps: int = 20,
                 num_categories: int = 1):
        super().__init__()

        if config is not None:
            embed_dim = getattr(config, "embed_dim", embed_dim)
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            action_dim = getattr(config, "action_dim", action_dim)
            horizon = getattr(config, "horizon", horizon)
            num_heads = getattr(config, "num_heads", num_heads)
            num_layers = getattr(config, "num_layers", num_layers)
            dropout = getattr(config, "dropout", dropout)
            num_inference_timesteps = getattr(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = getattr(config, "num_categories", num_categories)
            self.config = config
        else:
            from types import SimpleNamespace
            self.config = SimpleNamespace(embed_dim=embed_dim, hidden_dim=hidden_dim,
                                          action_dim=action_dim, horizon=horizon,
                                          num_heads=num_heads, num_layers=num_layers,
                                          dropout=dropout, num_inference_timesteps=num_inference_timesteps,
                                          num_categories=num_categories)
        print(f"num_inference_timesteps {num_inference_timesteps}")
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = config.per_action_dim
        self.action_dim = config.action_dim
        self.use_adaflow = getattr(self.config, "use_adaflow", False)
        self.adaflow_eta = getattr(self.config, "adaflow_eta", 0.5)
        self.adaflow_min_steps = getattr(self.config, "adaflow_min_steps", 2)
        self.adaflow_max_steps = getattr(self.config, "adaflow_max_steps", num_inference_timesteps)


        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(embed_dim=embed_dim, num_heads=num_heads,
                                   hidden_dim=embed_dim*4, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(embed_dim)
        self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim)

        self.mlp_head = CategorySpecificMLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                                            output_dim=action_dim, num_categories=num_categories)

        self.state_encoder = None
        if hasattr(self.config, "state_dim") and self.config.state_dim is not None:
            state_hidden = getattr(self.config, "state_hidden_dim", embed_dim)

            self.state_encoder = CategorySpecificMLP(input_dim=self.config.state_dim,
                                                    hidden_dim=state_hidden,
                                                    output_dim=embed_dim,
                                                    num_categories=num_categories)

        self.action_encoder = None
        if horizon > 1:
            per_action_dim = getattr(self.config, "per_action_dim", None)
            if per_action_dim is None:
                per_action_dim = action_dim // horizon if action_dim % horizon == 0 else action_dim
            self.action_encoder = MultiEmbodimentActionEncoder(action_dim=per_action_dim,
                                                               embed_dim=embed_dim,
                                                               hidden_dim=embed_dim,  
                                                               horizon=horizon,
                                                               num_categories=num_categories)
        self.variance_head = AdaFlowVarianceHead(embed_dim) if self.use_adaflow else None
        self._cached_x_pooled = None

    def freeze_base_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("variance_head."):
                param.requires_grad = False

    def unfreeze_variance_head(self):
        if self.variance_head is None:
            return
        for param in self.variance_head.parameters():
            param.requires_grad = True

    def _build_context_tokens(self, fused_tokens: torch.Tensor, state: torch.Tensor, embodiment_id: torch.LongTensor):
        context_tokens = fused_tokens
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1)
            context_tokens = torch.cat([context_tokens, state_emb], dim=1)
        return context_tokens

    def _get_time_embedding(self, t_val, batch_size: int, device: torch.device):
        if torch.is_tensor(t_val):
            t_tensor = t_val.to(device=device, dtype=self.dtype).view(batch_size)
        else:
            t_tensor = torch.full((batch_size,), float(t_val), device=device, dtype=self.dtype)
        time_index = (t_tensor * 1000).clamp(0, 999).long()
        return self.time_pos_enc(1000)[:, time_index, :].squeeze(0)

    def _pool_action_tokens(self, x: torch.Tensor):
        B = x.shape[0]
        if self.horizon > 1:
            x_flat = x.reshape(B, -1)
            return self.seq_pool_proj(x_flat)
        return x.squeeze(1)

    def forward(self, fused_tokens: torch.Tensor, state: torch.Tensor = None,
                actions_gt: torch.Tensor = None, embodiment_id: torch.LongTensor = None, 
                action_mask: torch.Tensor = None,
                ): 

        if actions_gt is None:
            return self.get_action(fused_tokens, state=state, embodiment_id=embodiment_id)
        B = fused_tokens.size(0)
        device = fused_tokens.device

        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens = self._build_context_tokens(fused_tokens, state, embodiment_id)

        t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device).to(dtype=self.dtype)
        time_emb = self._get_time_embedding(t, B, device)

        actions_gt_seq = actions_gt  
        noise = torch.rand_like(actions_gt) * 2 - 1

        if action_mask is not None:
            action_mask = action_mask.to(dtype=noise.dtype, device=noise.device)
            assert action_mask.shape == noise.shape, f"action_mask shape {action_mask.shape} != noise shape {noise.shape}"
            noise = noise * action_mask

        if self.horizon > 1:
            noise_seq = noise.view(B, self.horizon, self.per_action_dim)
        else:
            noise_seq = noise.unsqueeze(1)

        if self.horizon > 1:
            t_broadcast = t.view(B, 1, 1)
        else:
            t_broadcast = t.view(B, 1)
        action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq


        if self.horizon > 1 and self.action_encoder is not None:
            action_tokens = self.action_encoder(action_intermediate_seq, embodiment_id)  
        else:
            if not hasattr(self, "single_action_proj"):
                self.single_action_proj = nn.Linear(self.per_action_dim, self.embed_dim).to(device)
            action_tokens = self.single_action_proj(action_intermediate_seq) 

        x = action_tokens  
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)

        x = self.norm_out(x)  
        x_pooled = self._pool_action_tokens(x)
        self._cached_x_pooled = x_pooled

        pred_velocity = self.mlp_head(x_pooled, embodiment_id) 
        if self.variance_head is None:
            return pred_velocity, noise
        log_sqrt_var = self.variance_head(x_pooled)
        return pred_velocity, noise, log_sqrt_var

    def eval_velocity(self, action, t_val, context_tokens, embodiment_id, action_mask_seq, per_action_dim):
        B = action.shape[0]
        device = action.device
        time_emb = self._get_time_embedding(t_val, B, device)

        if self.horizon > 1:
            action_seq = action.view(B, self.horizon, per_action_dim)
        else:
            action_seq = action.view(B, 1, per_action_dim)

        if self.horizon > 1 and self.action_encoder is not None:
            action_seq = action_seq * action_mask_seq
            action_tokens = self.action_encoder(action_seq, embodiment_id)
        else:
            if hasattr(self, "single_action_proj"):
                action_tokens = self.single_action_proj(action_seq)
            else:
                self.single_action_proj = nn.Linear(per_action_dim, self.embed_dim).to(device)
                action_tokens = self.single_action_proj(action_seq)

        x = action_tokens
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)
        x = self.norm_out(x)
        x_pooled = self._pool_action_tokens(x)
        self._cached_x_pooled = x_pooled

        pred_velocity = self.mlp_head(x_pooled, embodiment_id)
        return pred_velocity

    def get_action(self, fused_tokens: torch.Tensor, state: torch.Tensor = None, 
                   embodiment_id: torch.LongTensor = None, action_mask: torch.Tensor = None, 
                   verbose: bool = False,
                   steps: int = None,
                   solver: str = "euler",
                   dt_probe: float = 0.5):

        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None: embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
        context_tokens = self._build_context_tokens(fused_tokens, state, embodiment_id)

        action_dim_total = getattr(self.config, "action_dim", self.action_dim)
        if self.horizon > 1: per_action_dim = getattr(self.config, "per_action_dim", action_dim_total // self.horizon)
        else: per_action_dim = action_dim_total

        action = (torch.rand(B, action_dim_total, device=device) * 2 - 1)

        action_mask_seq = None

        if action_mask is not None:
            expected_full_size = B * self.horizon * per_action_dim

            if action_mask.numel() == expected_full_size:
                action_mask_seq = action_mask.view(B, self.horizon, per_action_dim)
            else:
                action_mask_seq = action_mask.view(B, 1, per_action_dim).repeat(1, self.horizon, 1)

            action_mask_seq = action_mask_seq.to(dtype=action.dtype, device=device)

        if not hasattr(self, "single_action_proj") and (self.horizon == 1 or self.action_encoder is None):
            self.single_action_proj = nn.Linear(per_action_dim, self.embed_dim).to(device)

        if solver == "adaflow":
            if self.variance_head is None:
                raise ValueError("solver='adaflow' requires --use_adaflow so that variance_head is available.")
            return adaflow_sample(
                model=self,
                action=action,
                context_tokens=context_tokens,
                embodiment_id=embodiment_id,
                action_mask_seq=action_mask_seq,
                per_action_dim=per_action_dim,
                eta=self.adaflow_eta,
                min_steps=self.adaflow_min_steps,
                max_steps=self.adaflow_max_steps,
                verbose=verbose,
            )

        if solver == "rk45":
            rtol, atol = 1e-3, 1e-5
            t, dt = 0.0, 0.1
            step_count = 0

            a21 = 1/5
            a31, a32 = 3/40, 9/40
            a41, a42, a43 = 44/45, -56/15, 32/9
            a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
            a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
            b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
            e1, e3, e4, e5, e6, e7 = 71/57600, -71/16695, 71/1920, -17253/339200, 22/525, -1/40

            v_fn = lambda x_val, t_val: self.eval_velocity(x_val, t_val, context_tokens, embodiment_id, action_mask_seq, per_action_dim)

            k1 = v_fn(action, t)
            while t < 1.0 - 1e-5:
                if t + dt > 1.0:
                    dt = 1.0 - t

                k2 = v_fn(action + dt * (a21 * k1), t + 0.2 * dt)
                k3 = v_fn(action + dt * (a31 * k1 + a32 * k2), t + 0.3 * dt)
                k4 = v_fn(action + dt * (a41 * k1 + a42 * k2 + a43 * k3), t + 0.8 * dt)
                k5 = v_fn(action + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), t + (8/9) * dt)
                k6 = v_fn(action + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), t + dt)

                action_next = action + dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
                k7 = v_fn(action_next, t + dt)

                err_vec = dt * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)
                scale = atol + rtol * torch.max(torch.abs(action), torch.abs(action_next))
                err = torch.max(torch.abs(err_vec) / scale).item()

                dt_next = dt * max(0.2, min(5.0, 0.9 * (err + 1e-8)**(-0.2)))

                if err <= 1.0:
                    action = action_next
                    t += dt
                    k1 = k7  
                    step_count += 1

                dt = dt_next

            metadata = {"steps": step_count, "sim": 0.0, "mag": 0.0}
            return action, metadata

        if solver == "dpm_multistep":
            target_steps = steps if steps is not None else 10
            dt = 1.0 / target_steps
            t = 0.0
            v_prev = None

            for i in range(target_steps):
                v_current = self.eval_velocity(action, t, context_tokens, embodiment_id, action_mask_seq, per_action_dim)
                if i == 0:
                    action = action + v_current * dt
                else:
                    action = action + (1.5 * v_current - 0.5 * v_prev) * dt
                v_prev = v_current
                t += dt

            metadata = {"steps": target_steps, "sim": 0.0, "mag": 0.0}
            return action, metadata

        if solver == "heun":
            target_steps = steps if steps is not None else 10
            dt = 1.0 / target_steps
            t = 0.0

            for i in range(target_steps):
                v1 = self.eval_velocity(action, t, context_tokens, embodiment_id, action_mask_seq, per_action_dim)
                x_mid = action + v1 * dt
                v2 = self.eval_velocity(x_mid, t + dt, context_tokens, embodiment_id, action_mask_seq, per_action_dim)
                action = action + 0.5 * (v1 + v2) * dt
                t += dt

            metadata = {"steps": target_steps * 2, "sim": 0.0, "mag": 0.0}
            return action, metadata
        if steps is None:

            t = 0.0

            v_start = self.eval_velocity(action, t, context_tokens, embodiment_id, action_mask_seq, per_action_dim)
            x_mid = action + v_start * dt_probe
            v_mid = self.eval_velocity(x_mid, t + dt_probe, context_tokens, embodiment_id, action_mask_seq, per_action_dim)

            flat_v_start = v_start.view(B, -1)
            flat_v_mid = v_mid.view(B, -1)

            cos_sim = F.cosine_similarity(flat_v_start, flat_v_mid, dim=1)
            sim_score = cos_sim.min().item()

            mag_start = flat_v_start.norm(dim=1)
            mag_mid = flat_v_mid.norm(dim=1)
            mag_ratio = torch.minimum(mag_start, mag_mid) / (torch.maximum(mag_start, mag_mid) + 1e-6)
            mag_score = mag_ratio.min().item()

            epsilon = 0.008 
            curvature = 1.0 - sim_score

            raw_steps = 2 + 2 * math.floor(curvature / epsilon)
            target_steps = int(min(max(raw_steps, 2), 20))

            if verbose:
                print(f"[Adaptive] Sim: {sim_score:.4f} (Err: {curvature:.4f}) -> Steps: {target_steps}")

        else:
            target_steps = steps
            t = 0.0
            v_start = self.eval_velocity(action, t, context_tokens, embodiment_id, action_mask_seq, per_action_dim)

            sim_score = 0.0
            mag_score = 0.0


        if steps is None and target_steps == 2:
            action = x_mid + v_mid * (1.0 - dt_probe)
        else:
            dt = 1.0 / target_steps
            action = action + v_start * dt
            t += dt

            for _ in range(target_steps - 1):
                v = self.eval_velocity(action, t, context_tokens, embodiment_id, action_mask_seq, per_action_dim)
                action = action + v * dt
                t += dt

        metadata = {
            "steps": target_steps,
            "sim": sim_score,
            "mag": mag_score 
        }

        return action, metadata
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
