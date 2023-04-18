import numpy as np
import matplotlib.pyplot as plt

def load_multiple_energies(models, seed_range):
    energies_list = {model: [] for model in models}
    for model in models:
        for seed in seed_range:
            energy_file = f"{model}/energy-{seed}.txt"
            energies = np.loadtxt(energy_file)
            energies_list[model].append(energies)
        energies_list[model] = np.array(energies_list[model]).mean(axis=0)
    return energies_list

# def plot_energies_comparison(models, seed_range):
#     energies_dict = load_multiple_energies(models, seed_range)
#     plt.figure()

#     markers = ['o', 'x', 's']
#     for i, (model, energies) in enumerate(energies_dict.items()):
#         plt.scatter(range(len(energies)), energies, label=f"{model} Energies", marker=markers[i])

#     plt.xlabel("Relaxation")
#     plt.ylabel("Energy")
#     plt.title("Energies comparison of three algorithms")
#     plt.legend()
#     plt.show()
    
def plot_energies_comparison(models, seed_range):
    energies_dict = load_multiple_energies(models, seed_range)
    plt.figure()

    markers = ['o', 'x', 's']
    cumulative_relaxation = 0
    for i, (model, energies) in enumerate(energies_dict.items()):
        plt.scatter(range(cumulative_relaxation, cumulative_relaxation + len(energies)), 
                    energies, label=f"{model}", marker=markers[i], s=5)
        
        # Add turning point vertical line for each model
        if i > 0:
            plt.axvline(x=cumulative_relaxation - 0.5, color='red', linestyle='--')
        
        cumulative_relaxation += len(energies)

    plt.xlabel("Relaxation")
    plt.ylabel("Energy")
    plt.title("Energies comparison of three algorithms")
    plt.legend()
    plt.show()


# Call the function with the models and seed range
models = ["rhn0", "rhns", "rhng"]
seed_range = range(0, 30)  # Adjust the seed range according to your data
plot_energies_comparison(models, seed_range)


def plot_energy_histograms(models, seed_range):
    energies_dict = load_multiple_energies(models, seed_range)
    plt.figure()

    for i, (model, energies) in enumerate(energies_dict.items()):
        plt.hist(energies, bins='auto', alpha=0.5, label=f"{model}")

    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.title("Histogram of energies for three algorithms")
    plt.legend()
    plt.show()

# Call the function with the models and seed range
models = ["rhn0", "rhns", "rhng"]
seed_range = range(0, 11)  # Adjust the seed range according to your data
plot_energy_histograms(models, seed_range)