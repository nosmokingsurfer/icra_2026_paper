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
    

def plot_all_grads(model: torch.nn.Module):
    for n, p in model.named_parameters():
        plot_grads_hist(model, n)
        

def plot_norm_grads(model: torch.nn.Module, epoch=None, grad_tensor=None, folder="/home/dmitry/Documents/research/grad_images"):
    norm_grad_dict = dict()
    for n, p in model.named_parameters():
        # norm_grad_dict[n] = torch.max(torch.abs(p.grad))
        norm_grad_dict[n] = torch.linalg.norm(p.grad)
        
    if grad_tensor is not None:
        norm_grad_dict["grad_tensor"] = torch.linalg.norm(grad_tensor)
    
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
        


def update_grads_dict(norm_grad_dict: dict, model: torch.nn.Module, grad_tensor=None):
    
    for n, p in model.named_parameters():
        if n in norm_grad_dict:
            norm_grad_dict[n].append(torch.linalg.norm(p.grad))
        else:
            norm_grad_dict[n] = [torch.linalg.norm(p.grad)]
        
    if grad_tensor is not None:        
        if "grad_tensor" in norm_grad_dict:
            norm_grad_dict["grad_tensor"].append(torch.linalg.norm(grad_tensor))
        else:
            norm_grad_dict["grad_tensor"] = [torch.linalg.norm(grad_tensor)]
        
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