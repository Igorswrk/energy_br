import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union


# plot functions
def plot_line_evolution_energy_month_per_x(x, y, hue, data, title, bbox_to_anchor=(0.5, -0.4), ncols=6):
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 5))
    sns.lineplot(x=x, y=y, hue=hue, data=data)
    
    plt.title(title)
    plt.xlabel('Mês')
    plt.ylabel('Geração de Energia (MWmed)')
    plt.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, ncols=ncols)
    plt.subplots_adjust(bottom=0.2)  
    
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.savefig(os.path.join(output_folder, f"{hue}.png"))
    
    plt.show()


def plot_general_view(df, num_lin, num_cols, figsize, hspace, column, column_values):
    fig, axs = plt.subplots(num_lin, num_cols, figsize=figsize)
    plt.subplots_adjust(hspace=hspace)
    
    for i, j in enumerate(column_values):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        data = df.loc[df[column] == j, 'val_geracao_med_month'].rolling(12).mean()
        ax.plot(data)
        ax.set_title(j)

    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.savefig(os.path.join(output_folder, f"{column}.png"))
    plt.show()

# LOSS functions
class WMAPE(torch.nn.Module):

    def __init__(self):
        super(WMAPE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = [""]
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        return y_hat.squeeze(-1)

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        if mask is None:
            mask = torch.ones_like(y_hat)

        num = mask * (y - y_hat).abs()
        den = mask * y.abs()
        return num.sum() / den.sum()

def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
    
