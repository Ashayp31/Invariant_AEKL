from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def calc_gradient_penalty(discriminator, data, generated_data, device, gp_coef=10):
    batch_size = data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, 1)
    epsilon = epsilon.expand_as(data)
    epsilon = epsilon.to(device)

    interpolation = epsilon * data + (1 - epsilon) * generated_data
    interpolation = Variable(interpolation, requires_grad=True)
    interpolation = interpolation.to(device)

    interpolation_logits, _ = discriminator(interpolation)
    grad_outputs = torch.ones(interpolation_logits.size())

    grad_outputs = grad_outputs.to(device)

    gradients = autograd.grad(outputs=interpolation_logits,
                              inputs=interpolation,
                              grad_outputs=grad_outputs,
                              create_graph=True,
                              retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return gp_coef * ((gradients_norm - 1) ** 2).mean()


def train_epoch_fanogan_stage1(
        generator,
        discriminator,
        loader,
        optimizer_g,
        scheduler_g,
        optimizer_d,
        scheduler_d,
        device,
        epoch,
        writer,
        gp_coeff,
):
    generator.train()
    discriminator.train()
    raw_generator = generator.module if hasattr(generator, "module") else generator

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        img = x["MRI"].to(device)
        # GENERATOR
        optimizer_g.zero_grad()

        z = Variable(torch.randn(img.shape[0], raw_generator.z_dim).to(device))

        fake_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        # Train on fake images`
        fake_validity, _ = discriminator(fake_imgs)

        #print("GEN")
        #print(fake_validity)
        g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        optimizer_g.step()
        scheduler_g.step()

        losses = OrderedDict(
            g_loss=g_loss,
        )

        # DISCRIMINATOR
        for i in range(5):
            optimizer_d.zero_grad()
            fake_imgs = generator(z)
            real_validity, _ = discriminator(img)
            #print(real_validity)
            fake_validity, _ = discriminator(fake_imgs)
            #print(fake_validity)
            gradient_penalty = calc_gradient_penalty(discriminator, img.data, fake_imgs.data, device, gp_coeff)
            print(real_validity)
            print(fake_validity)
            print(gradient_penalty)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
            print(d_loss)
            print()

            d_loss.backward()
            optimizer_d.step()
            scheduler_d.step()

        print()
        losses["d_loss"] = d_loss

        writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader) + step)
        writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "g_loss": f"{losses['g_loss'].item():.6f}",
                "d_loss": f"{losses['d_loss'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_g):.6f}"
            },
        )


def eval_fanogan_stage1(
        generator,
        discriminator,
        loader,
        device,
        step,
        writer
):
    generator.eval()
    total_losses = OrderedDict()
    raw_generator = generator.module if hasattr(generator, "module") else generator


    for x in loader:
        img = x["MRI"].to(device)

        with torch.no_grad():
            # GENERATOR
            z = Variable(torch.randn(img.shape[0], raw_generator.z_dim).to(device))
            fake_imgs = generator(z)
            fake_validity, _ = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            losses = OrderedDict(
                g_loss=g_loss,
            )

            # DISCRIMINATOR
            fake_imgs = generator(z)
            real_validity, _ = discriminator(img)
            fake_validity, _ = discriminator(fake_imgs)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

        losses["d_loss"] = d_loss

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * img.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    images = fake_imgs[:4].cpu().clamp(min=-1, max=1)
    images = images[:,:,:,168//2,:]
    grid_img = make_grid(images, nrow=4, normalize=True)
    writer.add_image('FAKE IMGS', grid_img, global_step=step)

    return total_losses['d_loss']


def train_fanogan_stage1(
        generator,
        discriminator,
        start_epoch,
        best_loss,
        train_loader,
        val_loader,
        optimizer_g,
        scheduler_g,
        optimizer_d,
        scheduler_d,
        n_epochs,
        eval_freq,
        writer_train,
        writer_val,
        device,
        run_dir,
        gp_coeff,
):
    raw_model = generator.module if hasattr(generator, "module") else generator
    raw_discriminator = discriminator.module if hasattr(discriminator, "module") else discriminator

    val_loss = eval_fanogan_stage1(
        generator=generator,
        discriminator=discriminator,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val
    )

    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):

        train_epoch_fanogan_stage1(
            generator=generator,
            discriminator=discriminator,
            loader=train_loader,
            optimizer_g=optimizer_g,
            scheduler_g=scheduler_g,
            optimizer_d=optimizer_d,
            scheduler_d=scheduler_d,
            device=device,
            epoch=epoch,
            writer=writer_train,
            gp_coeff=gp_coeff,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_fanogan_stage1(
                generator=generator,
                discriminator=discriminator,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                'best_loss': best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))
            torch.save(checkpoint, str(run_dir / f"checkpoint_{len(train_loader) * epoch}.pth"))

    print(f"Training finished!")
    print(f"Saving final generator...")
    torch.save(raw_model.state_dict(), str(run_dir / 'final_generator_model.pth'))
    torch.save(raw_discriminator.state_dict(), str(run_dir / 'final_discriminator_model.pth'))

    return val_loss


def train_epoch_fanogan_stage2(
        generator,
        encoder,
        discriminator,
        loader,
        optimizer_e,
        scheduler_e,
        optimizer_d,
        scheduler_d,
        device,
        epoch,
        writer,
):
    generator.eval()
    encoder.train()
    discriminator.train()

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        img = x["MRI"].to(device)
        optimizer_e.zero_grad()
        optimizer_d.zero_grad()
        generator.zero_grad()

        z = encoder(img)
        fake_imgs = generator(z)
        _, image_feats = discriminator(img)
        _, recon_feats = discriminator(fake_imgs)

        loss_img = F.mse_loss(img, fake_imgs)
        loss_feat = F.mse_loss(image_feats, recon_feats)

        e_loss = loss_img + loss_feat

        e_loss.backward()

        optimizer_e.step()
        scheduler_e.step()

        optimizer_d.step()
        scheduler_d.step()

        losses = OrderedDict(
            e_loss=e_loss,
            loss_img=loss_img,
            loss_feat=loss_feat,
        )

        writer.add_scalar("lr_g", get_lr(optimizer_e), epoch * len(loader) + step)
        writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "e_loss": f"{losses['e_loss'].item():.6f}",
                "loss_img": f"{losses['loss_img'].item():.6f}",
                "loss_feat": f"{losses['loss_feat'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_e):.6f}"
            },
        )


def eval_fanogan_stage2(
        generator,
        encoder,
        discriminator,
        loader,
        device,
        step,
        writer
):
    generator.eval()
    total_losses = OrderedDict()

    for x in loader:
        img = x["MRI"].to(device)

        with torch.no_grad():
            z = encoder(img)
            fake_imgs = generator(z)
            _, image_feats = discriminator(img)
            _, recon_feats = discriminator(fake_imgs)
            loss_img = F.mse_loss(img, fake_imgs)
            loss_feat = F.mse_loss(image_feats, recon_feats)
            e_loss = loss_img + loss_feat

        losses = OrderedDict(
            e_loss=e_loss,
            loss_img=loss_img,
            loss_feat=loss_feat,
        )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * img.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    return total_losses['e_loss']


def train_fanogan_stage2(
        generator,
        encoder,
        discriminator,
        start_epoch,
        best_loss,
        train_loader,
        val_loader,
        optimizer_e,
        scheduler_e,
        optimizer_d,
        scheduler_d,
        n_epochs,
        eval_freq,
        writer_train,
        writer_val,
        device,
        run_dir
):
    raw_encoder = encoder.module if hasattr(encoder, "module") else encoder
    raw_discriminator = discriminator.module if hasattr(discriminator, "module") else discriminator

    val_loss = eval_fanogan_stage2(
        generator=generator,
        encoder=encoder,
        discriminator=discriminator,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val
    )

    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):

        train_epoch_fanogan_stage2(
            generator=generator,
            encoder=encoder,
            discriminator=discriminator,
            loader=train_loader,
            optimizer_e=optimizer_e,
            scheduler_e=scheduler_e,
            optimizer_d=optimizer_d,
            scheduler_d=scheduler_d,
            device=device,
            epoch=epoch,
            writer=writer_train
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_fanogan_stage2(
                generator=generator,
                encoder=encoder,
                discriminator=discriminator,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_e': optimizer_e.state_dict(),
                "scheduler_e": scheduler_e.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                'best_loss': best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

    print(f"Training finished!")
    print(f"Saving final generator...")
    torch.save(raw_encoder.state_dict(), str(run_dir / 'final_encoder_model.pth'))
    torch.save(raw_discriminator.state_dict(), str(run_dir / 'final_discriminator_model.pth'))

    return val_loss
