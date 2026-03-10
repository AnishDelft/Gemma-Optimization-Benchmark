"""
Quantize Gemma 4B using NVIDIA Model-Optimizer (PTQ)
======================================================
Prerequisites:
    pip install nvidia-modelopt[torch] transformers datasets torch

References:
    https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq

Supported quantization formats:
    - fp8          → Hopper / Ada GPUs (H100, A100-like, RTX 4090+)
    - int4_awq     → Weight-only INT4, good for low-batch inference
    - int8_sq      → INT8 SmoothQuant (weights + activations)
    - w4a8_awq     → INT4 weights + INT8 activations (experimental)
    - nvfp4        → Blackwell GPUs only (B100/B200)
"""

import argparse
import copy
import os
import json
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

MODEL_ID     = "/workspace/gemma-3-4b-it"
EXPORT_DIR   = "/workspace/gemma-3-4b-it-fp8"
QUANT_FORMAT = "fp8"      # fp8 | int4_awq | int8_sq | w4a8_awq | nvfp4
CALIB_SIZE   = 512
MAX_SEQ_LEN  = 6000
DEVICE       = "cuda"

# ──────────────────────────────────────────────
# Vision-encoder prefixes to EXCLUDE from quant
# Gemma3 embeds SigLIP; text-only calibration
# cannot compute valid amaxes for these layers.
# ──────────────────────────────────────────────
VISION_PREFIXES = [
    "*vision_tower*",
    "*vision_model*",
    "*multi_modal_projector*",
    "*siglip*",
    "*image_newline*",
]

BASE_CONFIGS = {
    "fp8":      mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "int8_sq":  mtq.INT8_SMOOTHQUANT_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "nvfp4":    mtq.NVFP4_DEFAULT_CFG,
}


def _build_quant_config(base_cfg: dict) -> dict:
    """
    Deep-copy the base config and inject disable rules for every
    vision-encoder pattern so those layers stay in bf16.
    """
    cfg  = copy.deepcopy(base_cfg)
    qcfg = cfg.setdefault("quant_cfg", {})
    for prefix in VISION_PREFIXES:
        qcfg[f"{prefix}weight_quantizer"] = {"enable": False}
        qcfg[f"{prefix}input_quantizer"]  = {"enable": False}
    return cfg


# ──────────────────────────────────────────────
# Patch modelopt export to handle None amax
# ──────────────────────────────────────────────

def _patch_export_none_amax():
    """
    Monkey-patch unified_export_hf._export_quantized_weight so it skips
    weight quantizers whose _amax / amax is None instead of crashing with
    'NoneType' object has no attribute 'to'.
    Root cause: vision-encoder quantizers have no calibrated amax because
    they were never exercised during text-only calibration.
    """
    import modelopt.torch.export.unified_export_hf as _ueh
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

    _orig = _ueh._export_quantized_weight

    def _safe_export(sub_module, dtype):
        for m in sub_module.modules():
            if not isinstance(m, TensorQuantizer):
                continue
            # Resolve the actual amax tensor (try both attribute names)
            amax_val = getattr(m, "amax", None)
            if amax_val is None:
                amax_val = m._buffers.get("_amax", None)

            if amax_val is None:
                # No calibration data → disable quantizer so export skips it
                m.disable()
            else:
                # Ensure _amax buffer exists and is float32 (what export expects)
                if "_amax" not in m._buffers or m._buffers["_amax"] is None:
                    m.register_buffer("_amax", amax_val.clone().to(torch.float32))
                elif m._buffers["_amax"].dtype != torch.float32:
                    m._buffers["_amax"] = m._buffers["_amax"].to(torch.float32)

        return _orig(sub_module, dtype)

    _ueh._export_quantized_weight = _safe_export
    print("  Patched modelopt export to handle None amax (vision encoder layers)")


# ──────────────────────────────────────────────
# Steps
# ──────────────────────────────────────────────

def load_model(model_id: str):
    print(f"[1/4] Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def build_calib_dataloader(tokenizer, calib_size: int, max_seq_len: int):
    """Build calibration dataset from CNN/DailyMail (text-only, no images)."""
    print("[2/4] Building calibration dataset")
    texts = []

    for split in ("train", "validation"):
        if len(texts) >= calib_size:
            break
        try:
            ds = load_dataset("cnn_dailymail", "3.0.0", split=split, streaming=True)
            for sample in ds:
                texts.append(sample["article"])
                if len(texts) >= calib_size:
                    break
        except Exception as e:
            print(f"  Warning: Could not load cnn_dailymail/{split}: {e}")

    texts = texts[:calib_size]
    print(f"  Calibration samples collected: {len(texts)}")

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )

    return [
        {
            "input_ids":      encodings["input_ids"][i].unsqueeze(0).to(DEVICE),
            "attention_mask": encodings["attention_mask"][i].unsqueeze(0).to(DEVICE),
        }
        for i in range(len(texts))
    ]


def make_forward_loop(calib_dataset):
    def forward_loop(model):
        for batch in calib_dataset:
            with torch.no_grad():
                model(**batch)
    return forward_loop


def quantize_model(model, quant_format: str, forward_loop):
    print(f"[3/4] Quantizing: format={quant_format}  (vision encoder excluded)")
    if quant_format not in BASE_CONFIGS:
        raise ValueError(f"Unknown format '{quant_format}'. Choose from: {list(BASE_CONFIGS)}")
    quant_cfg = _build_quant_config(BASE_CONFIGS[quant_format])
    model = mtq.quantize(model, quant_cfg, forward_loop)
    print("  Quantization complete.")
    return model


def export_model(model, export_dir: str, quant_format: str):
    print(f"[4/4] Exporting checkpoint to: {export_dir}")
    _patch_export_none_amax()
    os.makedirs(export_dir, exist_ok=True)

    try:
        with torch.inference_mode():
            export_hf_checkpoint(model, export_dir=export_dir)
        print(f"HF checkpoint saved to: {export_dir}")
        print(f"Load with: AutoModelForCausalLM.from_pretrained('{export_dir}')")

    except Exception as e:
        print(f"HF export failed: {e}")
        print("Falling back to torch.save ...")
        save_path = os.path.join(export_dir, "quantized_model.pt")
        torch.save(model.state_dict(), save_path)
        with open(os.path.join(export_dir, "quant_info.json"), "w") as f:
            json.dump({"model_id": MODEL_ID, "quant_format": quant_format,
                       "note": "Raw state_dict; load with torch.load + model.load_state_dict"}, f, indent=2)
        print(f"State dict saved to: {save_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",     default=MODEL_ID)
    p.add_argument("--export_dir",   default=EXPORT_DIR)
    p.add_argument("--quant_format", default=QUANT_FORMAT, choices=list(BASE_CONFIGS))
    p.add_argument("--calib_size",   type=int, default=CALIB_SIZE)
    p.add_argument("--max_seq_len",  type=int, default=MAX_SEQ_LEN)
    return p.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model_id)
    calib_dataset    = build_calib_dataloader(tokenizer, args.calib_size, args.max_seq_len)
    forward_loop     = make_forward_loop(calib_dataset)
    model            = quantize_model(model, args.quant_format, forward_loop)
    export_model(model, args.export_dir, args.quant_format)
    print(f"\n Done!  vllm serve {args.export_dir} --quantization modelopt")


if __name__ == "__main__":
    main()
