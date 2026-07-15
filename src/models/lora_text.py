import re
from contextlib import contextmanager
from peft import LoraConfig, get_peft_model


def apply_text_lora(model, r=16, alpha=32, dropout=0.05):
    # 1) freeze everything first
    for p in model.parameters():
        p.requires_grad = False
    # 2) wrap the text tower; PeftModel forwards attribute access so `.bert` still resolves
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=["query", "value"], bias="none")
    model.text_encoder = get_peft_model(model.text_encoder, cfg)
    # 3) eval mode => frozen BatchNorm/dropout in text_proj stay deterministic; LoRA still gets grad
    model.eval()
    return model


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@contextmanager
def teacher_ctx(model):
    with model.text_encoder.disable_adapter():
        yield


def joint_target_modules(model, proj_heads='none'):
    """Explicit full module names so peft matches exactly (no layer.0/layer.10 collision)."""
    import torch.nn as nn
    names = []
    text_re = re.compile(r'text_encoder\.bert\.encoder\.layer\.([0-5])\.attention\.self\.(query|value)$')
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name.startswith('vision_encoder') and name.endswith('attn.qkv'):
            names.append(name)
        elif text_re.search(name):
            names.append(name)
    if proj_heads in ('vision_only', 'both'):
        names.append('vision_proj.2')          # Linear inside Sequential(BN, Dropout, Linear)
    if proj_heads == 'both':
        names.append('text_proj.2')
    return names


def apply_joint_lora(model, r=16, alpha=32, dropout=0.05, proj_heads='none'):
    for p in model.parameters():
        p.requires_grad = False
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=joint_target_modules(model, proj_heads), bias='none')
    model = get_peft_model(model, cfg)
    model.eval()        # frozen BN/dropout; LoRA still trains
    return model


def itm_target_modules(model):
    """Stage-2 ITM: BERT fusion layers 6-11, BOTH self-attention and cross-attention q/v.
    (cross-attention exists only on fusion layers >= fusion_layer=6.)"""
    import torch.nn as nn
    pat = re.compile(r'text_encoder\.bert\.encoder\.layer\.(6|7|8|9|10|11)\.'
                     r'(attention|crossattention)\.self\.(query|value)$')
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and pat.search(n)]


def apply_itm_lora(model, r=16, alpha=32, dropout=0.05):
    """LoRA on BERT 6-11 self+cross q/v; itm_head FULL fine-tuned (modules_to_save -> trainable
    AND persisted by save_pretrained). Apply AFTER merging the frozen Stage-1 ITC adapter."""
    for p in model.parameters():
        p.requires_grad = False
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=itm_target_modules(model), bias='none',
                     modules_to_save=['itm_head'])
    model = get_peft_model(model, cfg)
    model.eval()        # frozen BN/dropout; LoRA + itm_head still train
    return model
