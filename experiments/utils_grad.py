import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
from pathlib import Path

def plot_grads_hist(model: torch.nn.Module,
                      module_name: str):

    grad = dict(model.named_parameters())[module_name].grad
    plt.title(module_name)
    plt.hist(grad.flatten().detach().cpu())
    plt.savefig(f"{module_name}.png")
    plt.close()


def plot_all_grads(model: torch.nn.Module, writer=None, step=None):
    for n, p in model.named_parameters():
        if writer is None:
            plot_grads_hist(model, n)
        elif p.grad is not None:
            writer.add_histogram(f'grad_hist/{n}', p.grad, step)


def plot_norm_grads(model: torch.nn.Module, epoch=None, grad_tensor=None, folder="/home/dmitry/Documents/research/grad_images", writer=None, step=None):
    norm_grad_dict = dict()
    for n, p in model.named_parameters():
        norm_grad_dict[n] = torch.linalg.norm(p.grad)

    if grad_tensor is not None:
        norm_grad_dict["grad_tensor"] = torch.linalg.norm(grad_tensor)

    if writer is not None and step is not None:
        for n, v in norm_grad_dict.items():
            writer.add_scalar(f'grad_norms/{n.replace(".", "/")}', v.item(), step)
    else:
        plt.bar(torch.arange(len(norm_grad_dict)),
                norm_grad_dict.values(),
                tick_label=norm_grad_dict.keys())
        plt.xticks(rotation=90, fontsize=6)
        plt.ylim(0.01, 40)
        plt.yscale("log")
        if epoch is not None:
            plt.ylabel(f"Norm grad distribution, epoch {epoch}")
        else:
            plt.ylabel("Norm grad distribution")
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', dpi=200)
        plt.close()
        buf.seek(0)
        if epoch is not None:
            Image.open(buf).rotate(-90, expand=True).save(Path(folder) / f"norm_grads_epoch_{epoch}.png")
        else:
            Image.open(buf).rotate(-90, expand=True).save(Path(folder) / "norm_grads.png")



def update_grads_dict(norm_grad_dict: dict, model: torch.nn.Module, grad_tensor=None, writer=None, step=None):

    for n, p in model.named_parameters():
        norm = torch.linalg.norm(p.grad)
        if n in norm_grad_dict:
            norm_grad_dict[n].append(norm)
        else:
            norm_grad_dict[n] = [norm]
        if writer is not None and step is not None:
            writer.add_scalar(f'grad_norm/{n}', norm.item(), step)

    if grad_tensor is not None:
        norm = torch.linalg.norm(grad_tensor)
        if "grad_tensor" in norm_grad_dict:
            norm_grad_dict["grad_tensor"].append(norm)
        else:
            norm_grad_dict["grad_tensor"] = [norm]
        if writer is not None and step is not None:
            writer.add_scalar('grad_norm/grad_tensor', norm.item(), step)

    return norm_grad_dict


def plot_grads_dict(norm_grad_dict, folder="/home/dmitry/Documents/research/grad_images"):
    for n in norm_grad_dict:
        plt.figure()
        plt.title(f"{n} grads norm")
        plt.xlabel("step")
        plt.yscale("log")
        plt.plot(norm_grad_dict[n])
        plt.savefig(Path(folder) / f"module_{n}_grads_norm.png")
        plt.close()
