import matplotlib.pyplot as plt 
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for all text rendering
        "font.family": "serif",  # Use a serif font like Computer Modern
        "font.serif": ["Computer Modern Roman"],  # Default LaTeX font
        "axes.labelsize": 11,  # Axis label font size
        "axes.titlesize": 14,  # Title font size
        "legend.fontsize": 7,  # Legend font size
        "xtick.labelsize": 10,  # X-axis tick labels
        "ytick.labelsize": 10,  # Y-axis tick labels
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",  # Extra LaTeX packages
    }
)

def plot_energy_minimization_in_sweep(dmrg_energies: list[float], reference : float):


    N_steps = len(dmrg_energies)
    steps = np.arange(N_steps)

    dmrg_energies = np.array(dmrg_energies) - reference

    fig = plt.figure(figsize=(3,3))
    plt.plot(steps,dmrg_energies)
    plt.xlabel("DMRG Steps")
    plt.ylabel("Energy above Reference")
    plt.xlim(left =0, right = N_steps)
    plt.yscale("log")

    return fig