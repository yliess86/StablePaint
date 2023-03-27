from __future__ import annotations

from argparse import ArgumentParser
from PIL import Image
from stablepaint.data import (HintMethods, SketchDataset)
from stablepaint.nn import (DDIM, DDPM, Decoder, DiagonalGaussianDistribution, Encoder, latent_linear_beta_schedule, LPIPSWithDiscriminator, UNet)
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import (cycle, to_pil, WarmupCosineDecayLR)

import numpy as np
import os
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.set_detect_anomaly(mode=False)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


parser = ArgumentParser()
parser.add_argument("--stage",       type=str,   required=True)

parser.add_argument("--steps",       type=int,   default=100_000)
parser.add_argument("--accumulate",  type=int,   default=1)
parser.add_argument("--batch_size",  type=int,   default=8)
parser.add_argument("--log_every",   type=int,   default=100)

parser.add_argument("--name",        type=str,   default="pixiv")
parser.add_argument("--sketch_size", type=int,   default=512)
parser.add_argument("--crop_size",   type=int,   default=256)

parser.add_argument("--device",      type=str,   default="cuda")
parser.add_argument("--dtype",       type=str,   default="bfloat16")
parser.add_argument("--t",           type=float, default=1.0)

parser.add_argument("--scale",       type=float, default=0.18215)
parser.add_argument("--corruption",  type=float, default=0.5)
parser.add_argument("--lineart",     type=str,   default=None)
parser.add_argument("--hints",       type=str,   default=None)
args = parser.parse_args()

num_workers = os.cpu_count() // 2
device, dtype = args.device, eval(f"torch.{args.dtype}")
to = lambda x: x.to(device=device, dtype=dtype)

interpolate = lambda x, scale_factor: torch.nn.functional.avg_pool2d(x, (scale_factor, scale_factor))


if args.stage in ["illustration", "lineart"]:
    if args.stage == "illustration": hint_method = HintMethods.IDENTITY
    if args.stage == "lineart"     : hint_method = HintMethods.mix(HintMethods.IDENTITY, HintMethods.RANDOM, p=0.2)

    dataset = SketchDataset(f"dataset/{args.name}", args.sketch_size, args.crop_size, hint_method, train=True, jobs=num_workers)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    iter_batch = cycle(loader)
    
    if args.stage == "illustration": in_channels = 4
    if args.stage == "lineart"     : in_channels = 5

    encoder: Encoder = to(Encoder(in_channels, t=args.t))
    decoder: Decoder = to(Decoder(t=args.t))
    criterion: LPIPSWithDiscriminator = to(LPIPSWithDiscriminator(decoder[-1].weight, start_disc=args.steps // 4, t=args.t))

    if args.stage == "lineart":
        print(f"Loading Illustration Autoencoder: logs/ckpts/{args.name}_illustration_[encoder|decoder].pt...")
        encoder.load(torch.load(f"logs/ckpts/{args.name}_illustration_encoder.pt"))
        decoder.load(torch.load(f"logs/ckpts/{args.name}_illustration_decoder.pt"))

    if args.stage == "illustration": parameters = list(encoder.parameters()) + list(decoder.parameters())
    if args.stage == "lineart"     : parameters = list(encoder.parameters())

    optim_autoencoder = AdamW(parameters, lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-2)
    optim_discriminator = AdamW(criterion.disc.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-2)
    lr_scheduler = WarmupCosineDecayLR(optim_autoencoder, lr=1e-3, min_lr=1e-3 / 10, warmup_steps=int(5e-3 * args.steps), decay_steps=args.steps)

    pbar = tqdm(range(args.steps), desc="Train")
    for step in pbar:
        loss_autoencoder, loss_discriminator = 0, 0
        for _ in range(args.accumulate):
            x, l, h = map(to, next(iter_batch))

            if args.stage == "illustration": model_input = torch.cat((x, l), dim=1)
            if args.stage == "lineart"     : model_input = torch.cat((h, l), dim=1)

            post = DiagonalGaussianDistribution(encoder(model_input))
            reco = decoder(post.sample()).clip_(-1.0, 1.0)
            micro_loss_autoencoder, micro_loss_discriminator = criterion(x, reco, post, step=step)
            
            micro_loss_autoencoder /= args.accumulate; micro_loss_autoencoder.backward()
            micro_loss_discriminator /= args.accumulate; micro_loss_discriminator.backward()
            
            loss_autoencoder += micro_loss_autoencoder.item()
            loss_discriminator += micro_loss_discriminator.item()

        lr_scheduler.update(step)
        optim_autoencoder.step()
        optim_autoencoder.zero_grad(set_to_none=True)
        optim_discriminator.step()
        optim_discriminator.zero_grad(set_to_none=True)
        
        pbar.set_postfix(loss_autoencoder=f"{loss_autoencoder:.2e}", loss_discriminator=f"{loss_discriminator:.2e}", lr=f"{lr_scheduler.compute(step):.2e}")

        if step % args.log_every == 0:
            print("Saving Images...")
            with torch.inference_mode():
                to_pil(x).save(f"logs/imgs/{args.name}_{args.stage}_autoencoder_target.png")
                to_pil(decoder(DiagonalGaussianDistribution(encoder(model_input)).sample())).save(f"logs/imgs/{args.name}_{args.stage}_autoencoder_reconstruction.png")

            print("Saving Checkpoint...")
            torch.save(encoder.state_dict(), f"logs/ckpts/{args.name}_{args.stage}_encoder.pt")
            torch.save(decoder.state_dict(), f"logs/ckpts/{args.name}_{args.stage}_decoder.pt")


if args.stage == "scale":
    dataset = SketchDataset(f"dataset/{args.name}", args.sketch_size, args.crop_size, HintMethods.IDENTITY, train=True, jobs=num_workers)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    iter_batch = iter(loader)

    with torch.inference_mode():
        encoder = to(Encoder(in_channels=4, t=args.t)).load(torch.load(f"logs/ckpts/{args.name}_illustration_encoder.pt"))
        x, l, h = map(to, next(iter_batch))
        scale = 1.0 / DiagonalGaussianDistribution(encoder(torch.cat((x, l), dim=1))).sample().std()

    print("latent scale factor:", round(scale.item(), 4))


if args.stage == "noise_model":
    print(f"Loading Autoencoders: logs/ckpts/{args.name}_illustration_[encoder|decoder].pt and logs/ckpts/{args.name}_lineart_[encoder].pt...")
    x_encoder: Encoder = to(Encoder(in_channels=4, t=args.t)).load(torch.load(f"logs/ckpts/{args.name}_illustration_encoder.pt"))
    l_encoder: Encoder = to(Encoder(in_channels=5, t=args.t)).load(torch.load(f"logs/ckpts/{args.name}_lineart_encoder.pt"))
    decoder: Decoder = to(Decoder(t=args.t)).load(torch.load(f"logs/ckpts/{args.name}_illustration_decoder.pt"))

    hint_method = HintMethods.mix(HintMethods.IDENTITY, HintMethods.RANDOM, p=0.2)
    dataset = SketchDataset(f"dataset/{args.name}", args.sketch_size, args.crop_size, hint_method, train=True, jobs=num_workers)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    iter_batch = cycle(loader)

    unet: UNet = to(UNet(t=args.t))
    ddm: DDPM = to(DDPM(1_000, latent_linear_beta_schedule))

    optim = AdamW(unet.parameters(), lr=1e-3)
    lr_scheduler = WarmupCosineDecayLR(optim, lr=1e-4, min_lr=1e-4 / 10, warmup_steps=int(5e-4 * args.steps), decay_steps=args.steps)

    def denoise(z_t: Tensor, t: Tensor, ctx: Tensor, start: int) -> Tensor:
        for i in tqdm(range(start, 1, -1), desc="Denoise"):
            t.fill_(i); z_t = ddm.backward_diffusion(unet(z_t, t.to(dtype=dtype), ctx), z_t, t)
        return z_t

    loss_history, lr_history = [], []
    t = torch.randint(0, len(ddm) + 1, (args.batch_size,), device=device, dtype=torch.long)
    pbar = tqdm(range(args.steps), desc="Train")
    for step in pbar:
        loss = 0.0
        for _ in range(args.accumulate):
            x, l, h = map(to, next(iter_batch))

            with torch.no_grad():
                z_x = args.scale * DiagonalGaussianDistribution(x_encoder(torch.cat((x, l), dim=1))).sample()
                z_l = args.scale * DiagonalGaussianDistribution(l_encoder(torch.cat((h, l), dim=1))).sample()
                t, n = t.random_(0, len(ddm) + 1), torch.randn_like(z_x)
                z_n = ddm.forward_diffusion(z_x, t, n)
                ctx = torch.cat((z_l, interpolate(h, scale_factor=8)), dim=1)
            
            micro_loss = F.mse_loss(unet(z_n, t.to(dtype=dtype), ctx), n, reduction="mean")
            micro_loss /= args.accumulate; micro_loss.backward()

            loss += micro_loss.item()

        torch.nn.utils.clip_grad_norm_(unet.parameters(), 0.5)
        lr_scheduler.update(step)
        optim.step()
        optim.zero_grad(set_to_none=True)
        loss_history.append(loss)
        lr_history.append(lr_scheduler.compute(step))

        pbar.set_postfix(loss=f"{loss:.2e}", lr=f"{lr_scheduler.compute(step):.2e}")

        if step % args.log_every == 0:
            select = lambda x: x[:min(32, len(x))]

            print("Saving Image...")
            with torch.inference_mode():
                to_pil(torch.cat((
                    decoder(denoise(ddm.forward_diffusion(select(z_l), select(t).fill_(int(0.750 * len(ddm)))), select(t), select(ctx), int(0.750 * len(ddm))) / args.scale),
                    decoder(denoise(ddm.forward_diffusion(select(z_l), select(t).fill_(int(0.500 * len(ddm)))), select(t), select(ctx), int(0.500 * len(ddm))) / args.scale),
                    decoder(denoise(ddm.forward_diffusion(select(z_l), select(t).fill_(int(0.250 * len(ddm)))), select(t), select(ctx), int(0.250 * len(ddm))) / args.scale),
                    decoder(denoise(ddm.forward_diffusion(select(z_l), select(t).fill_(int(0.125 * len(ddm)))), select(t), select(ctx), int(0.125 * len(ddm))) / args.scale),
                ), dim=0)).save(f"logs/imgs/{args.name}_latent_noise_model.png")
            
            print("Saving Checkpoint...")
            torch.save(unet.state_dict(), f"logs/ckpts/{args.name}_noise_model.pt")

            print("Ploting History...")
            import matplotlib.pyplot as plt
            plt.subplot(2, 1, 1)
            plt.plot(loss_history)
            plt.subplot(2, 1, 2)
            plt.plot(lr_history)
            plt.draw()
            plt.savefig("history.png")
            

if args.stage == "demo":
    with torch.inference_mode():
        encoder: Encoder = to(Encoder(in_channels=5, t=args.t)).load(torch.load(f"logs/ckpts/{args.name}_lineart_encoder.pt"))
        decoder: Decoder = to(Decoder(t=args.t)).load(torch.load(f"logs/ckpts/{args.name}_illustration_decoder.pt"))
        unet: UNet = to(UNet(t=args.t)).load(torch.load(f"logs/ckpts/{args.name}_noise_model.pt"))
        ddm: DDPM = to(DDPM(1_000, latent_linear_beta_schedule))

        def denoise(z_t: Tensor, t: Tensor, ctx: Tensor, start: int) -> Tensor:
            for i in tqdm(range(start, 1, -1), desc="Denoise"):
                t.fill_(i); z_t = ddm.backward_diffusion(unet(z_t, t.to(dtype=dtype), ctx), z_t, t)
            return z_t

        lineart = Image.open(args.lineart).convert("L")
        l = F.interpolate(torch.from_numpy(np.array(lineart))[None, None], (args.crop_size, args.crop_size))
        l = to((2.0 * l / 255.0 - 1.0).repeat(args.batch_size, 1, 1, 1))
        
        hints = Image.open(args.hints).convert("RGBA")
        h = F.interpolate(torch.from_numpy(np.array(hints)).permute(2, 0, 1)[None], (args.crop_size, args.crop_size))
        h = to((h / 255.0).repeat(args.batch_size, 1, 1, 1))

        z_l = args.scale * DiagonalGaussianDistribution(encoder(torch.cat((h, l), dim=1))).sample()
        ctx = torch.cat((z_l, interpolate(h, scale_factor=8)), dim=1)

        corruption = int(args.corruption * len(ddm))
        t = torch.full((args.batch_size,), corruption, device=device, dtype=torch.long)
        to_pil(decoder(denoise(ddm.forward_diffusion(z_l, t), t, ctx, corruption) / args.scale)).show()