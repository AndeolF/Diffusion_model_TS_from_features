import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, welch

# Paramètres de base
fs = 1000  # Fréquence d'échantillonnage (Hz)
duration = 0.5  # Durée du signal en secondes
f_c = 150  # Fréquence de coupure du filtre (Hz)
numtaps = 31  # Nombre de taps du filtre

# Génération de bruit blanc
np.random.seed(0)
n_samples = int(duration * fs)
white_noise = np.random.randn(n_samples)


# Fonctions de filtrage
def apply_filter(signal, window_type):
    if isinstance(window_type, tuple):
        kernel = firwin(numtaps=numtaps, cutoff=f_c, fs=fs, window=window_type)
    else:
        kernel = firwin(numtaps=numtaps, cutoff=f_c, fs=fs, window=window_type)
    return lfilter(kernel, 1.0, signal)


# Liste des fenêtres à tester
windows = {
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Kaiser (β=8.6)": ("kaiser", 8.6),
    "Kaiser (β=5)": ("kaiser", 5),
    "Kaiser (β=3)": ("kaiser", 3),
    "Kaiser (β=1)": ("kaiser", 1),
}

# Tracé des résultats
plt.figure(figsize=(10, 6))
for name, window in windows.items():
    filtered = apply_filter(white_noise, window)
    f, Pxx = welch(filtered, fs=fs, nperseg=1024)
    plt.semilogy(f, Pxx, label=name)

# Ajoute aussi la PSD du bruit blanc d'origine
f_orig, Pxx_orig = welch(white_noise, fs=fs, nperseg=1024)
plt.semilogy(f_orig, Pxx_orig, "k--", label="Bruit blanc (non filtré)")

plt.title("Comparaison des filtres avec différentes fenêtres")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Densité spectrale (échelle log)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
