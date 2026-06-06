

import math
from einops import repeat
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm
from models.GBC import GBC, BottConv
from models.PAF import PAF

class SAVSS_2D(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_size=7, bias=False, conv_bias=False, init_layer_scale=None, default_hw_shape=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.n_directions = 8 

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = BottConv(in_channels=self.d_inner, out_channels=self.d_inner, mid_channels=self.d_inner // 16, kernel_size=3, padding=1, stride=1)
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant": nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else: nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad(): self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

    def sass(self, hw_shape):
        """
        完全体 SASS 扫描策略：平行蛇形 + 对角蛇形 (共 8 条路径)
        """
        H, W = hw_shape
        L = H * W
        
        o1, o2, o3, o4, o5, o6, o7, o8 = [], [], [], [], [], [], [], []
        d1, d2, d3, d4, d5, d6, d7, d8 = [], [], [], [], [], [], [], []
        inv = [[-1]*L for _ in range(8)]
        
        # 修复方向编码逻辑：语义方向标签物理反转
        reverse_dir = {1: 3, 3: 1, 2: 4, 4: 2, 5: 6, 6: 5, 7: 8, 8: 7}
        
        # 1. 平行蛇形 (行优先, 正向)
        for i in range(H):
            if i % 2 == 0:
                for j in range(W):
                    idx = i * W + j
                    inv[0][idx] = len(o1)
                    o1.append(idx)
                    d1.append(1) 
            else:
                for j in range(W):
                    idx = i * W + (W - 1 - j)
                    inv[0][idx] = len(o1)
                    o1.append(idx)
                    d1.append(3) 
        
        # 2. 平行蛇形 (行优先, 反向)
        o2 = o1[::-1]
        for idx_seq, orig_idx in enumerate(o2): inv[1][orig_idx] = idx_seq
        d2 = [reverse_dir[d] for d in d1[::-1]] 

        # 3. 平行蛇形 (列优先, 正向)
        for j in range(W):
            if j % 2 == 0:
                for i in range(H):
                    idx = i * W + j
                    inv[2][idx] = len(o3)
                    o3.append(idx)
                    d3.append(2) 
            else:
                for i in range(H):
                    idx = (H - 1 - i) * W + j
                    inv[2][idx] = len(o3)
                    o3.append(idx)
                    d3.append(4) 
                    
        # 4. 平行蛇形 (列优先, 反向)
        o4 = o3[::-1]
        for idx_seq, orig_idx in enumerate(o4): inv[3][orig_idx] = idx_seq
        d4 = [reverse_dir[d] for d in d3[::-1]]

        # 5. 主对角蛇形 (正向)
        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + j
                        inv[4][idx] = len(o5)
                        o5.append(idx)
                        d5.append(5) 
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + j
                        inv[4][idx] = len(o5)
                        o5.append(idx)
                        d5.append(6) 
        
        # 6. 主对角蛇形 (反向)
        o6 = o5[::-1]
        for idx_seq, orig_idx in enumerate(o6): inv[5][orig_idx] = idx_seq
        d6 = [reverse_dir[d] for d in d5[::-1]]

        # 7. 副对角蛇形 (正向)
        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + (W - j - 1)
                        inv[6][idx] = len(o7)
                        o7.append(idx)
                        d7.append(7) 
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + (W - j - 1)
                        inv[6][idx] = len(o7)
                        o7.append(idx)
                        d7.append(8) 

        # 8. 副对角蛇形 (反向)
        o8 = o7[::-1]
        for idx_seq, orig_idx in enumerate(o8): inv[7][orig_idx] = idx_seq
        d8 = [reverse_dir[d] for d in d7[::-1]] 

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4), tuple(o5), tuple(o6), tuple(o7), tuple(o8)), \
               (tuple(inv[0]), tuple(inv[1]), tuple(inv[2]), tuple(inv[3]), tuple(inv[4]), tuple(inv[5]), tuple(inv[6]), tuple(inv[7])), \
               (tuple(d1), tuple(d2), tuple(d3), tuple(d4), tuple(d5), tuple(d6), tuple(d7), tuple(d8))

    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        ssm_state = None
        xz = self.in_proj(x)
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=-1)
        
        # 局部卷积增强 
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        # 参数投影
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        # 获取包含了动态方向序列的完整 8 路径
        orders, inverse_orders, directions = self.sass(hw_shape)
        
        direction_Bs = [self.direction_Bs[list(d), :] for d in directions]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in direction_Bs]

        # 🌟 修复 1：将 dt, B, C 参数同步按照扫描路径重排，实现真正的结构序列对齐
        y_scan = []
        for o, inv_order, dB in zip(orders, inverse_orders, direction_Bs):
            o_idx = list(o)

            # 同步重排序列与状态参数
            x_seq = x_conv[:, o_idx, :].permute(0, 2, 1).contiguous()
            dt_seq = dt[:, :, o_idx].contiguous()
            B_seq = B[:, :, o_idx].contiguous()
            C_seq = C[:, :, o_idx].contiguous()

            y_i = selective_scan_fn(
                x_seq,
                dt_seq,
                A,
                (B_seq + dB).contiguous(),
                C_seq,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_order, :]

            y_scan.append(y_i)

        y = sum(y_scan) * self.act(z)

        out = self.out_proj(y)
        if hasattr(self, 'gamma'): out = out * self.gamma
        return out

class SAVSS_Layer(nn.Module):
    def __init__(self, embed_dims, use_rms_norm, with_dwconv, drop_path_rate, mamba_cfg):
        super(SAVSS_Layer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})
        self.norm = RMSNorm(embed_dims) if use_rms_norm else nn.LayerNorm(embed_dims)
        self.with_dwconv = with_dwconv
        self.SAVSS_2D = SAVSS_2D(**mamba_cfg)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        self.linear_256 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.GN_256 = nn.GroupNorm(num_channels=256, num_groups=16)
        self.GBC_C = GBC(embed_dims)
        self.PAF_256 = PAF(embed_dims, embed_dims // 2)

    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape # 严格遵循传入的宽高
        
        x_in = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        for i in range(2): x_in = self.GBC_C(x_in)
        x_gbc = x_in.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        mixed_x = self.drop_path(self.SAVSS_2D(self.norm(x_gbc), hw_shape))
        
        b, l, c = mixed_x.shape
        h, w = hw_shape 
        
        paf_out = self.PAF_256(x_gbc.permute(0, 2, 1).reshape(b, c, h, w), mixed_x.permute(0, 2, 1).reshape(b, c, h, w))
        
        mixed_x_paf = self.GN_256(paf_out).reshape(b, c, h * w).permute(0, 2, 1)
        
        if self.with_dwconv:
            mixed_x_paf = mixed_x_paf.reshape(b, h, w, c).permute(0, 3, 1, 2)
            mixed_x_paf = self.GBC_C(mixed_x_paf).reshape(b, c, h * w).permute(0, 2, 1)

        res = self.linear_256(self.GN_256(mixed_x_paf.permute(0, 2, 1)).permute(0, 2, 1))
        return mixed_x + res