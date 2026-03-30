# ============================================================
# AeroGuard — TCN Model Definition
#
# Yeh file production TCN model define karti hai.
# Exact same architecture jo Colab mein train hui thi.
#
# Usage:
#   from src.models.tcn_model import load_tcn_model
#   model = load_tcn_model('artifacts/best_tcn.pt')
# ============================================================

# --- Standard library imports ---
#Numpy
import numpy as np
# torch: PyTorch ka core framework — tensors, autograd, GPU support sab yahan se aata hai
import torch
# torch.nn: Neural network layers, loss functions, aur module base class
import torch.nn as nn
# json: production_config.json padhne ke liye — threshold, channels wagerah store hoti hai yahan
import json
# os: file existence check (os.path.exists) ke liye — model ya config missing ho toh early fail karo
import os
# logger: loguru-based custom logger — har major step pe structured log entry dalta hai
from src.logger import logger
# ModelPredictionException: custom exception class — raw Python exceptions ko wrap karta hai
# context string ke saath taaki error traceback mein pata chale kahan fail hua
from src.exception import ModelPredictionException


# ============================================================
# TCN ARCHITECTURE — Exact same as Colab
# ============================================================
# NOTE: Is section ka code bilkul wahi hai jo Colab training mein use hua tha.
# Koi bhi architectural change yahan bhi karna padega aur Colab notebook mein bhi —
# warna state_dict load_state_dict() pe mismatch karega aur weights load nahi honge.


class CausalConv1d(nn.Module):
    """
    Causal dilated 1D convolution.
    Causal = future timesteps use nahi karta.
    Dilated = receptive field exponentially badhta hai.
    """
    # WHY CAUSAL?
    # Normal Conv1d symmetric padding use karta hai — yani left aur right dono taraf se context leta hai.
    # Aviation sensor data mein yeh problem hai: real-time inference mein future data available nahi hota.
    # CausalConv sirf past aur present dekhta hai — production-safe hai.
    #
    # WHY DILATED?
    # Dilation = gaps between kernel elements.
    # dilation=1  → normal conv, receptive field = kernel_size
    # dilation=2  → har do element ek skip, receptive field = 2*(kernel_size-1)+1
    # dilation=4  → aur bada receptive field... 
    # 8 layers mein 1,2,4,8,16,32,64,128 dilations → receptive field ~768 timesteps (4096 ke andar kaafi hai)
    # Matlab: ek single flight ke shuru ke patterns aur end ke patterns dono capture hote hain.

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        # Causal padding calculate karo: (kernel_size - 1) * dilation
        # Yeh exactly itna left-pad karta hai ki output ka har timestep
        # sirf apne left side (past) ke inputs dekhe, right side (future) nahi.
        self.padding = (kernel_size - 1) * dilation
        # Conv1d layer — bias=False hai kyunki BatchNorm baad mein aata hai
        # BatchNorm apna bias/shift khud seekhta hai, isliye Conv ka bias redundant hoga.
        self.conv    = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=self.padding,   # puri padding left side pe effectively — trim baad mein hoga
            dilation=dilation,
            bias=False
        )
        # BatchNorm1d: output channels ko normalize karta hai
        # Training stable rehti hai, learning rate sensitivity kam hoti hai
        self.bn      = nn.BatchNorm1d(out_channels)
        # Dropout: regularization — random neurons ko zero karta hai training mein
        # Overfitting rokta hai; inference pe automatically off hota hai (model.eval() se)
        self.dropout = nn.Dropout(dropout)
        # ReLU: activation function — negative values zero kar do, positive rakhlo
        # Computationally cheap aur TCN papers mein standard choice hai
        self.relu    = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, in_channels, Time)
        # Conv apply karo — output mein extra timesteps aa jayenge padding ki wajah se
        out = self.conv(x)
        # CAUSALITY ENFORCEMENT:
        # Conv ne self.padding extra timesteps right side pe generate kiye hain.
        # Unhe trim karo taaki output length == input length rahe.
        # Agar padding=0 hai (kernel_size=1 case) toh trim ki zaroorat nahi.
        out = out[:, :, :-self.padding] \
            if self.padding > 0 else out
        # BatchNorm → ReLU → Dropout ka order yahan:
        # BN pehle normalize karta hai, phir ReLU activate karta hai, phir dropout regularize karta hai.
        # Yeh "pre-activation" style nahi hai — standard post-conv BN hai.
        return self.dropout(self.relu(self.bn(out)))


class TCNBlock(nn.Module):
    """
    TCN Block = 2 CausalConv1d + Residual connection.
    """
    # WHY TWO CONVOLUTIONS PER BLOCK?
    # Har block mein 2 CausalConv1d stack hoti hain — same dilation dono mein.
    # Pehli conv: low-level features extract karo (edges, transitions)
    # Doosri conv: unhe refine karo (higher-level temporal patterns)
    # ResNet-style: 2 conv per block empirically better performance deta hai single conv se.
    #
    # WHY RESIDUAL CONNECTION?
    # Gradient flow problem: deep networks mein gradients vanish ho jaate hain backprop mein.
    # Residual (skip) connection direct path deta hai gradients ko earlier layers tak pahunchne ka.
    # 8 TCNBlocks deep network ke liye yeh critical hai.

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        # Pehli causal conv: in_channels → out_channels (channel dimension change hoti hai yahan)
        self.conv1 = CausalConv1d(
            in_channels, out_channels,
            kernel_size, dilation, dropout
        )
        # Doosri causal conv: out_channels → out_channels (channel dim same rehti hai)
        self.conv2 = CausalConv1d(
            out_channels, out_channels,
            kernel_size, dilation, dropout
        )
        # RESIDUAL CONNECTION:
        # Agar in_channels != out_channels, toh direct addition possible nahi (dimension mismatch).
        # Is case mein 1x1 Conv + BN se channels match karo — "projection shortcut" kehte hain isse.
        # Agar channels same hain (block ke andar ke layers), toh Identity (no-op) use karo.
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=1, bias=False),  # 1x1 conv = channel mixer, no temporal mixing
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        # Final ReLU residual addition ke baad — main path + skip path ka sum activate karo
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, in_channels, Time)

        # Residual (skip) path — input directly shortcut se jaata hai
        res = self.residual(x)

        # Main path: conv1 → conv2
        out = self.conv1(x)
        out = self.conv2(out)

        # TEMPORAL ALIGNMENT:
        # Theoretically dono same length hone chahiye, lekin floating point aur padding
        # edge cases mein 1 timestep ka difference aa sakta hai.
        # min_len se dono ko trim karo taaki addition possible ho.
        min_len = min(out.shape[-1], res.shape[-1])
        out = out[..., :min_len]
        res = res[..., :min_len]

        # Residual addition + ReLU — yahi TCN ka core hai
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network — AeroGuard version.
    Input  : (B, 31, 4096)
    Output : (B, 1)
    Dilation: 1, 2, 4, 8, 16, 32, 64, 128
    """
    # ARCHITECTURE OVERVIEW:
    # ┌─────────────────────────────────────────────────────┐
    # │  Input: (Batch, 31 channels, 4096 timesteps)        │
    # │     ↓                                               │
    # │  TCNBlock (dilation=1)   — local patterns           │
    # │  TCNBlock (dilation=2)   — 2x longer context        │
    # │  TCNBlock (dilation=4)   — 4x longer context        │
    # │  TCNBlock (dilation=8)                              │
    # │  TCNBlock (dilation=16)                             │
    # │  TCNBlock (dilation=32)                             │
    # │  TCNBlock (dilation=64)                             │
    # │  TCNBlock (dilation=128) — flight-level patterns    │
    # │     ↓                                               │
    # │  Global Average Pooling (mean over time dim)        │
    # │     ↓                                               │
    # │  Dropout                                            │
    # │     ↓                                               │
    # │  Linear(64 → 64) → ReLU → Dropout → Linear(64 → 1) │
    # │     ↓                                               │
    # │  Output: (Batch, 1) — raw logit (NOT sigmoid yet)   │
    # └─────────────────────────────────────────────────────┘
    #
    # WHY 31 CHANNELS?
    # 23 raw NGAFID sensors + 8 physics-informed engineered features
    # (e.g., power loading, engine stress, rate-of-change features)
    #
    # WHY 4096 TIMESTEPS?
    # Fixed-length windowing decision from data pipeline — flights padded/truncated to 4096.
    # Power of 2 hai — GPU memory access patterns ke liye efficient.

    def __init__(self, n_channels=31, n_filters=64,
                 kernel_size=3, n_layers=8, dropout=0.1):
        super().__init__()

        # Dilation schedule: exponential — 2^0, 2^1, ..., 2^(n_layers-1)
        # 8 layers ke liye: [1, 2, 4, 8, 16, 32, 64, 128]
        # Total receptive field ≈ 2 * (1+2+4+...+128) * (kernel_size-1) = 2 * 255 * 2 = 1020 timesteps
        # Matlab ek neuron 4096 mein se ~1000 timesteps ka context dekh sakta hai.
        dilations   = [2 ** i for i in range(n_layers)]
        # ModuleList: PyTorch ko pata chale ki yeh sub-modules hain
        # Regular Python list use karein toh parameters register nahi honge — state_dict mein nahi ayenge
        self.blocks = nn.ModuleList()
        # Pehle block ka input = raw sensor channels (31)
        # Baad ke blocks ka input = previous block ka output (n_filters=64)
        in_ch = n_channels

        for dilation in dilations:
            # Har dilation ke liye ek TCNBlock banao
            # Pehle block: 31 → 64 channels (in_ch = n_channels)
            # Baaki sab: 64 → 64 channels (in_ch = n_filters)
            self.blocks.append(
                TCNBlock(
                    in_ch, n_filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            # Pehle block ke baad in_ch update ho jaata hai n_filters pe
            in_ch = n_filters

        # Final dropout — classifier se pehle regularization
        self.dropout    = nn.Dropout(dropout)

        # Classification head:
        # Global avg pool ke baad (Batch, 64) tensor aata hai
        # Linear(64→64): non-linear feature mixing
        # ReLU: activate
        # Dropout: regularize
        # Linear(64→1): single logit output — binary classification ke liye
        # NOTE: Sigmoid yahan nahi hai — BCEWithLogitsLoss training mein numerically stable hai
        # Inference mein torch.sigmoid() manually apply karte hain (predict_single_flight mein)
        self.classifier = nn.Sequential(
            nn.Linear(n_filters, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (Batch, 31, 4096) — channels first format (PyTorch Conv1d convention)

        # Saare TCN blocks sequentially apply karo
        # x transforms: (B,31,4096) → (B,64,4096) → (B,64,4096) → ... (8 blocks)
        for block in self.blocks:
            x = block(x)

        # Global Average Pooling: time dimension (dim=-1) ka mean lo
        # (B, 64, 4096) → (B, 64)
        # WHY MEAN? Max pooling bhi option tha, lekin mean pooling
        # poore flight ka aggregate behavior capture karta hai — zyada robust hai anomaly detection mein
        x = x.mean(dim=-1)

        # Dropout before classifier
        x = self.dropout(x)

        # Classification head: (B, 64) → (B, 1)
        # Output: raw logit (no sigmoid) — BCEWithLogitsLoss ke liye
        return self.classifier(x)


# ============================================================
# MODEL LOADER
# ============================================================
# Yeh section production deployment ke liye hai.
# Training code yahan nahi hai — sirf load aur inference.
# Do public functions expose hote hain:
#   1. load_tcn_model()       — disk se model + config load karo
#   2. predict_single_flight() — ek flight array deke prediction lo


def load_tcn_model(
    model_path: str = 'artifacts/best_tcn.pt',
    config_path: str = 'artifacts/production_config.json',
    device: str = 'cpu'
) -> tuple[TCN, dict]:
    """
    Trained TCN model load karta hai.

    Args:
        model_path  : path to best_tcn.pt
        config_path : path to production_config.json
        device      : 'cpu' ya 'cuda'

    Returns:
        tuple: (model, config)
    """
    try:
        logger.info("TCN model load ho raha hai...")

        # ── STEP 1: Config load ──────────────────────────────────────
        # production_config.json mein yeh fields hoti hain (typically):
        #   - threshold  : classification boundary (e.g., 0.45)
        #   - n_channels : input channels (31)
        #   - n_timesteps: input timesteps (4096)
        # Config pehle load karo — model instantiation iske values pe depend karti hai
        if not os.path.exists(config_path):
            # Fail fast: config nahi hai toh model bana bhi nahi sakte sahi se
            raise FileNotFoundError(
                f"Production config nahi mila: {config_path}"
            )
        with open(config_path, 'r') as f:
            config = json.load(f)

        logger.info(f"Config loaded — "
                    f"threshold: {config['threshold']}, "
                    f"channels: {config['n_channels']}")

        # ── STEP 2: Model instantiate ────────────────────────────────
        # Architecture config se le lo — hardcode mat karo
        # Taaki agar kabhi n_channels ya n_layers change ho,
        # sirf config update karna padega, yeh code nahi
        model = TCN(
            n_channels  = config['n_channels'],
            n_filters   = 64,          # Colab mein hardcoded tha, same rakhna zaroori hai
            kernel_size = 3,           # Colab mein hardcoded tha
            n_layers    = 8,           # Colab mein hardcoded tha — dilation schedule same rahega
            dropout     = 0.1,         # Eval mode mein dropout off ho jaata hai automatically
        )

        # ── STEP 3: Weights load ─────────────────────────────────────
        if not os.path.exists(model_path):
            # Fail fast: .pt file nahi hai toh random weights se predict karna dangerous hai
            raise FileNotFoundError(
                f"Model weights nahi mile: {model_path}"
            )

        # torch.load: serialized state_dict read karta hai
        # map_location=device: GPU pe save hua model CPU pe load ho sakta hai (aur vice versa)
        # weights_only=True: SECURITY — arbitrary Python code execute hone se rokta hai
        #                    PyTorch 2.0+ mein recommended practice hai
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
        # state_dict ke parameter names aur shapes exactly match karne chahiye model ke saath
        # Agar architecture change hua toh yahan RuntimeError aayegi — intentional fail-fast behavior
        model.load_state_dict(state_dict)
        # eval() mode: dropout aur batchnorm inference behavior mein switch ho jaate hain
        #   - Dropout: neurons drop karna band, sab active
        #   - BatchNorm: running statistics use karta hai (training statistics nahi)
        model.eval()
        # Model ko requested device pe move karo (CPU ya CUDA GPU)
        model.to(device)

        # ── STEP 4: Sanity log ───────────────────────────────────────
        # Parameter count log karo — debugging ke liye useful
        # Expected: ~200K-500K parameters (lightweight model hai TCN)
        n_params = sum(
            p.numel() for p in model.parameters()
        )
        logger.info(f"TCN loaded — "
                    f"{n_params:,} parameters | "
                    f"device: {device}")

        # Dono return karo — caller ko config bhi chahiye (threshold inference mein lagta hai)
        return model, config

    except Exception as e:
        # Koi bhi exception (FileNotFoundError, RuntimeError, etc.) ko
        # ModelPredictionException mein wrap karo context ke saath
        # Taaki upstream caller (FastAPI endpoint etc.) ek consistent exception type handle kare
        raise ModelPredictionException(
            e, context="Loading TCN model"
        )


def predict_single_flight(
    model: TCN,
    flight_array: 'np.ndarray',
    config: dict,
    device: str = 'cpu'
) -> dict:
    """
    Ek flight ke liye prediction karta hai.

    Args:
        model        : loaded TCN model
        flight_array : (4096, 31) numpy array
        config       : production config dict
        device       : 'cpu' ya 'cuda'

    Returns:
        dict:
            probability  : float (0-1)
            prediction   : int (0=safe, 1=at-risk)
            severity     : str (NORMAL/MEDIUM/HIGH/CRITICAL)
            threshold    : float
    """
    try:
        # numpy import yahan kiya hai (lazy import) taaki module-level import se
        # circular dependency ya unnecessary overhead na ho agar numpy unavailable ho
        import numpy as np

        # ── STEP 1: Input validation ─────────────────────────────────
        # Pipeline se aane wala array exactly (4096, 31) hona chahiye
        # NOTE: PyTorch Conv1d (B, C, T) format expect karta hai
        #       lekin humara pipeline (T, C) = (4096, 31) format mein deta hai
        #       Transpose baad mein hoga
        expected = (config['n_timesteps'], config['n_channels'])
        if flight_array.shape != expected:
            # Shape mismatch = ya toh pipeline bug hai ya galat data aaya — hard fail karo
            raise ValueError(
                f"Expected shape {expected}, "
                f"got {flight_array.shape}"
            )

        # ── STEP 2: Tensor conversion ────────────────────────────────
        # (4096, 31) → transpose → (31, 4096) → newaxis → (1, 31, 4096)
        # Breakdown:
        #   flight_array.T       : (31, 4096)  — channels first
        #   [np.newaxis, ...]    : (1, 31, 4096) — batch dimension add karo
        #   torch.tensor(...)    : numpy → PyTorch tensor
        #   dtype=float32        : model weights float32 mein hain — match karo
        #   .to(device)          : CPU ya GPU pe move karo
        x = torch.tensor(
            flight_array.T[np.newaxis, ...],
            dtype=torch.float32
        ).to(device)

        # ── STEP 3: Inference ────────────────────────────────────────
        # torch.no_grad(): gradient computation band karo
        #   - Memory efficient hai (activations store nahi karta)
        #   - Faster hai (backward graph nahi banta)
        #   - Inference mein gradients ki zaroorat nahi hoti
        with torch.no_grad():
            # model(x): (1, 31, 4096) → (1, 1) raw logit
            # .squeeze(): (1, 1) → scalar tensor
            logit = model(x).squeeze()
            # sigmoid: logit → probability (0 to 1)
            # .item(): PyTorch scalar tensor → Python float
            prob  = torch.sigmoid(logit).item()

        # ── STEP 4: Thresholding ─────────────────────────────────────
        # Threshold config se aata hai — hardcode mat karo
        # Default usually 0.5 hota hai, lekin AeroGuard mein
        # class imbalance ke liye tune kiya gaya hai (e.g., 0.45)
        threshold = config['threshold']
        # Binary prediction: prob >= threshold → at-risk (1), warna safe (0)
        pred      = 1 if prob >= threshold else 0

        # ── STEP 5: Severity bucketing ───────────────────────────────
        # Probability ko 4 human-readable severity levels mein map karo
        # Yeh thresholds domain knowledge + stakeholder requirements pe based hain:
        #   CRITICAL (≥0.80): Immediate ground check required
        #   HIGH     (≥0.60): Flag for maintenance review
        #   MEDIUM   (≥0.40): Monitor closely on next flight
        #   NORMAL   (<0.40): No action needed
        # NOTE: Severity aur prediction alag hain —
        #       prob=0.55 → prediction=1 (at-risk, threshold=0.45) lekin severity=HIGH nahi, MEDIUM hai
        if prob >= 0.80:
            severity = "CRITICAL"
        elif prob >= 0.60:
            severity = "HIGH"
        elif prob >= 0.40:
            severity = "MEDIUM"
        else:
            severity = "NORMAL"

        # ── STEP 6: Return structured result ─────────────────────────
        # Dict format: downstream consumers (FastAPI, Streamlit, alert engine)
        # ke liye consistent interface — keys change mat karo
        return {
            'probability': round(prob, 4),  # 4 decimal places — display ke liye sufficient
            'prediction' : pred,             # 0 ya 1 — binary label
            'severity'   : severity,         # human-readable alert level
            'threshold'  : threshold,        # transparency ke liye return karo — caller ko pata chale kya use hua
        }

    except Exception as e:
        # Har exception wrap karo — consistent error handling poore pipeline mein
        raise ModelPredictionException(
            e, context="TCN single flight prediction"
        )