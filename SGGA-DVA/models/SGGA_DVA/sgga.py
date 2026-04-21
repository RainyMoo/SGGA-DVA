from ops.fc import FC, MLP
from ops.layer_norm import LayerNorm
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

def select_topk(
    grid: torch.Tensor,        # [B, G, d]
    rg_align: torch.Tensor,    # [B, R, G]
    rg_iou: torch.Tensor,      # [B, R, G]
    mask: torch.Tensor,        # [B, 1, 1, G]
    K: int
):
    """
    综合选择最重要的 Top-K grid，并同步 gather 对应的 align、iou 和 mask。

    Returns:
        topk_grid:   [B, K, d]
        topk_galign: [B, K, R]
        topk_giou:   [B, K, R]
        topk_gmask:  [B, 1, 1, K]
    """
    B, G, d = grid.shape
    _, R, _ = rg_align.shape

    # ============ 聚合得分 ==============
    score = rg_align
    
    grid_score = score.sum(dim=1)      # [B,G]

    # ============ 取 Top-K ==============
    K_use = min(K, G)
    topk_idx = torch.topk(grid_score, K_use, dim=-1).indices  # [B,K]

    batch_idx = torch.arange(B, device=grid.device)[:, None]

    # ============ Gather grid ============
    topk_grid = grid[batch_idx, topk_idx]  # [B,K,d]

    # ============ Gather rg_iou / rg_align ============
    # 先转置成 [B,G,R]，再在 dim=1 上 gather
    idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, R)  # [B,K,R]
    topk_giou = torch.gather(rg_iou.transpose(1, 2), dim=1, index=idx_expanded)   # [B,K,R]
    topk_galign = torch.gather(rg_align.transpose(1, 2), dim=1, index=idx_expanded)  # [B,K,R]

    # ============ Gather mask ============
    topk_idx_exp = topk_idx.view(B, 1, 1, K_use)
    topk_gmask = mask.gather(dim=-1, index=topk_idx_exp)  # [B,1,1,K]

    return topk_grid, topk_galign, topk_giou, topk_gmask

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    #def forward(self, v, k, q, mask):
    def forward(self, v, k, q, mask,rg_iou,rg_align):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask, rg_iou, rg_align)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask,rg_iou, rg_align):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        # if rg_iou is not None:
        #     scores = scores * rg_iou.unsqueeze(1)

        # if rg_align is not None:
        #     scores = scores.masked_fill(rg_align.unsqueeze(1)==0, -1e9)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask,rg_iou=None, rg_align=None)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y
    
class Merge_SGA(nn.Module):
    def __init__(self, __C):
        super(Merge_SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        
        
        self.ffn = FFN(__C)
        self.grid_topk = False
        self.region_topk = False

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout4 = nn.Dropout(__C.DROPOUT_R)
        self.norm4 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout5 = nn.Dropout(__C.DROPOUT_R)
        self.norm5 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout6 = nn.Dropout(__C.DROPOUT_R)
        self.norm6 = LayerNorm(__C.HIDDEN_SIZE)
        
    def forward(self, x, y, z, x_mask, y_mask, z_mask, rg_align, rg_iou):

        if self.grid_topk:
            y_topk, y_topk_align, y_topk_iou, y_new_mask= select_topk(y, rg_iou, rg_align, y_mask, K=64)

        else:
            y_topk = y
            y_new_mask = y_mask
            y_topk_iou = rg_iou.transpose(1,2)
            y_topk_align = rg_align.transpose(1,2)
            
        if self.region_topk:
            x_topk, x_topk_align, x_topk_iou, x_new_mask = select_topk(x,rg_iou.transpose(1, 2),rg_align.transpose(1, 2),x_mask,K=60)

        else:
            x_topk = x
            x_new_mask = x_mask
            x_topk_iou = rg_iou
            x_topk_align = rg_align
       
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=y_topk, k=y_topk, q=x, mask=y_new_mask,rg_iou=y_topk_iou.transpose(1,2),rg_align=y_topk_align.transpose(1,2))
        ))

        y = self.norm2(y + self.dropout2(
            self.mhatt1(v=x_topk, k=x_topk, q=y, mask=x_new_mask,rg_iou=x_topk_iou.transpose(1,2),rg_align=x_topk_align.transpose(1,2))
        ))
        
        x = self.norm3(x + self.dropout3(
            self.mhatt2(v=z, k=z, q=x, mask=z_mask, rg_iou=None, rg_align=None)
        ))
        
        y = self.norm4(y + self.dropout4(
            self.mhatt2(v=z, k=z, q=y, mask=z_mask, rg_iou=None, rg_align=None)
        ))
        
        x = self.norm5(x+ self.dropout5(
            self.ffn(x)
        ))

        y = self.norm6(y+ self.dropout6(
            self.ffn(y)
        ))
        

        return x, y


class SGGA_ED(nn.Module):
    def __init__(self, __C):
        super(SGGA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.union_enc_list = nn.ModuleList([Merge_SGA(__C) for _ in range(int(__C.LAYER))])
        

    def forward(self, lang, region, grid, lang_mask, region_mask, grid_mask, rg_align, rg_iou):
        # Get encoder last hidden vector
        
        for enc in self.enc_list:
            lang = enc(lang, lang_mask)

        for union_enc in self.union_enc_list:
            region, grid = union_enc(region, grid, lang, region_mask, grid_mask, lang_mask, rg_align, rg_iou)
        
        return lang, region, grid
    
    