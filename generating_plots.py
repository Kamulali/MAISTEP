import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

def load_data(file_path):
    return joblib.load(file_path)

def calculate_statistics(data, decimals):
    median = round(np.median(data), decimals)
    q1 = round(np.percentile(data, 16), decimals)
    q3 = round(np.percentile(data, 84), decimals)
    return median, q1, q3

def configure_plot(ax, color, xlabel):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=18)
    ax.tick_params(axis='both', which='major', direction='out', width=3, length=6, labelsize=16, pad=6)

def plot_data(ax, data, nbins, color, decimals, unit):
    hist, bin_edges, _ = ax.hist(data, bins=nbins, histtype='step', lw=3, alpha=0.9,      		    weights=np.ones_like(data) / len(data), color=color)
    median, q1, q3 = calculate_statistics(data, decimals)
    ymax = max(hist)
   # ax.axvline(median, color='gray', alpha=0.4, linestyle='-', ymax=ymax / ax.get_ylim()[1], linewidth=2)
   # ax.axvline(q1, color='orange', alpha=0.2, linestyle='dashed', ymax=ymax / ax.get_ylim()[1], linewidth=2)
   # ax.axvline(q3, color='orange', alpha=0.2, linestyle='dashed', ymax=ymax / ax.get_ylim()[1], linewidth=2)
    
    lower_bound = round(median - q1, decimals)
    upper_bound = round(q3 - median, decimals)
    ax.fill_betweenx([0, ymax], median - lower_bound, median + upper_bound, color='gray', alpha=0.2)
    
    median_text = f"${median:.{decimals}f}^{{+{upper_bound:.{decimals}f}}}_{{-{lower_bound:.{decimals}f}}}$ {unit}"
    ax.text(0.6, 1, median_text, transform=ax.transAxes, fontsize=18, color=color)

def analyze_data(directory_path, results_dir):
    files = [f for f in os.listdir(directory_path) if f.endswith('_saved_preds.txt')]
    
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        data = load_data(file_path)
        
        masses, radii, ages = data['mass'], data['radius'], data['age']
        data_list = [masses, radii, ages]
        colors = ['red', 'black', 'blue']
        decimals = [2, 3, 1]
        units = ['M$_{\odot}$', 'R$_{\odot}$', 'Gyr']
        labels = ['$M$ (M$_{\odot}$)', '$R$ (R$_{\odot}$)', r'$\tau$ (Gyr)']
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        nbins = np.int64(1 + np.ceil(np.log2(len(radii))))
        
        for ax, data, color, dec, unit, label in zip(axs, data_list, colors, decimals, units, labels):
            plot_data(ax, data, nbins, color, dec, unit)
            configure_plot(ax, color, label)
        
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'{file_name.replace("_saved_preds.txt", "")}.pdf')
        plt.savefig(save_path, bbox_inches='tight',dpi =2500)
        plt.close()
