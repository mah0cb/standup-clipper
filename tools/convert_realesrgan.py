"""realesr-general-x4v3.pth → ncnn (.param/.bin) dönüştürücü (bir kez çalıştırılır).

Resmi realesrgan-ncnn-vulkan paketinde bu model yok; yalnızca PyTorch ağırlığı yayınlanmış.
Kullanım: python tools/convert_realesrgan.py models/realesr-general-x4v3.pth models/realesrgan/models
Gerekli: pip install pnnx
"""
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """basicsr/archs/srvgg_arch.py ile aynı mimari (prelu, 64 kanal, 32 conv, x4)."""

    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def main(pth, out_dir, name="realesr-general-x4v3"):
    out_dir = Path(out_dir)
    m = SRVGGNetCompact()
    sd = torch.load(pth, map_location="cpu", weights_only=True)
    m.load_state_dict(sd.get("params", sd))
    m.eval()
    stem = name.replace("-", "_")  # pnnx çıktı adlarında "-" yerine "_" kullanır
    ts = out_dir / f"{stem}.pt"
    torch.jit.trace(m, torch.rand(1, 3, 64, 64)).save(str(ts))
    pnnx = Path(sys.executable).with_name("pnnx.exe")  # venv aktif olmasa da bulunur
    subprocess.run([str(pnnx) if pnnx.exists() else "pnnx", ts.name, "inputshape=[1,3,64,64]"], check=True, cwd=out_dir)
    # realesrgan-ncnn-vulkan giriş/çıkış blob adlarını "data" / "output" bekler
    param = (out_dir / f"{stem}.ncnn.param").read_text()
    lines = [" ".join("data" if t == "in0" else "output" if t == "out0" else t for t in ln.split(" "))
             for ln in param.splitlines()]
    (out_dir / f"{name}.param").write_text("\n".join(lines) + "\n")
    (out_dir / f"{stem}.ncnn.bin").replace(out_dir / f"{name}.bin")
    for f in out_dir.glob(f"{stem}*"):  # ara dosyalar
        f.unlink()
    print("hazır:", out_dir / f"{name}.param")


if __name__ == "__main__":
    main(*sys.argv[1:3])
