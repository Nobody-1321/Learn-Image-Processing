import numpy as np
import cv2
import lip

# Ejemplo de uso
if __name__ == "__main__":
    
    #img = cv2.imread("img_data/halftone_cat.webp", cv2.IMREAD_GRAYSCALE)
    #denoised, noise = lip.RemoveQuasiperiodicNoise(img, patch_size=200, threshold=2.5)
    
    #img = cv2.imread("img_data/mapper.jpg", cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    #denoised, noise = lip.RemoveQuasiperiodicNoise(img, patch_size=200, threshold=3.0, fmax=0.61)

    #img = cv2.imread("img_data/greenhead.webp")
    #img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    #denoised, noise = lip.RemoveQuasiperiodicNoiseBGR(img, patch_size=100, threshold=3.0, fmax=0.55)

    #img = cv2.imread("img_data/desert1.jpg")
    #img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    #denoised, noise = lip.RemoveQuasiperiodicNoiseBGR(img, patch_size=250, threshold=2.5, fmax=0.60)

    img = cv2.imread("img_data/monper.jpeg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (500, 500))
    denoised, noise = lip.RemoveQuasiperiodicNoise(img, patch_size=200, threshold=1.9, fmax=0.66)
    

    mag_o, phase_o = lip.ComputeFourierSpectra(img)
    mag_d, phase_d = lip.ComputeFourierSpectra(denoised)
    mag_n, phase_n = lip.ComputeFourierSpectra(noise)

    combined_mag = np.hstack((mag_o, mag_d, mag_n))
    combined = np.hstack((img, denoised, noise))

    cv2.imshow("Original | Denoised | Noise", combined)
    cv2.imshow("Fourier Magnitudes", combined_mag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt

def remove_quasiperiodic_noise(image, patch_size=128, threshold=3.0):
    # Preprocesamiento
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32) / 255.0
    height, width = image.shape
    
    # Ajuste de parámetros según el paper
    patch_size = min(patch_size, height, width)
    step = max(1, patch_size // 8)  # Paso de L/8 como en el paper
    f2 = 8 / patch_size  # Frecuencia mínima para detección de ruido

    
    # Extracción de parches con ventana de Hann
    hann_window = np.outer(np.hanning(patch_size), np.hanning(patch_size))
    patches = [
        image[y:y+patch_size, x:x+patch_size] * hann_window
        for y in range(0, height - patch_size, step)
        for x in range(0, width - patch_size, step)
    ]

    power_spectra = [np.abs(fftshift(fft2(patch)))**2 for patch in patches]    
    # Convertir la lista de espectros de potencia en un numpy array
    power_spectra = np.array(power_spectra)
    # Calcular el espectro de potencia promedio (media geométrica)
    avg_power_spectrum = np.exp(np.mean(np.log(power_spectra + 1e-10), axis=0))

    # Frecuencias radiales
    fy = np.fft.fftfreq(patch_size)[:, np.newaxis]
    fx = np.fft.fftfreq(patch_size)
    f = np.sqrt(fx**2 + fy**2)
    
    valid_mask = (f > f2/4) & (f < 0.655)
    
    log_f = np.log(f[valid_mask]).reshape(-1, 1)
    log_P = np.log(avg_power_spectrum[valid_mask]).ravel()

    # Ajuste robusto de la ley de potencia
    model = HuberRegressor()
    model.fit(log_f, log_P)
    log_P_pred = model.predict(log_f)
    alpha = -model.coef_[0]
    A = np.exp(model.intercept_)

    # Detección de outliers (excluyendo frecuencias bajas f < f2)
    residuals = log_P - log_P_pred
    std_res = np.std(residuals)
    upper_bound = log_P_pred + threshold * std_res
    outliers = (log_P > upper_bound) & (f[valid_mask].ravel() >= f2)

    # Mapa de outliers con simetría (cambio clave)
    outlier_mask = np.zeros_like(avg_power_spectrum, dtype=bool)
    outlier_mask[valid_mask] = outliers
    outlier_mask |= np.flip(outlier_mask, axis=0)  # Simetría vertical
    outlier_mask |= np.flip(outlier_mask, axis=1)  # Simetría horizontal

    # Redimensionar y suavizar (MANTENIDO)
    outlier_map_resized = cv2.resize(
        outlier_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )

    outlier_map_smoothed = gaussian_filter(outlier_map_resized, sigma=2.0)

    # Protección de la componente DC (MANTENIDO)
    cy, cx = height // 2, width // 2
    outlier_map_smoothed[cy-1:cy+2, cx-1:cx+2] = 0.0

    # Filtrado notch
    fft_image = fftshift(fft2(image))
    fft_filtered = fft_image * (1 - outlier_map_smoothed)
    denoised_image = np.real(ifft2(ifftshift(fft_filtered)))
    noise_component = image - denoised_image

    # Normalización y salida (MANTENIDO)
    denoised_image = np.clip(denoised_image, 0, 1)
    noise_component = (noise_component - noise_component.min()) / (noise_component.max() - noise_component.min())
    
    return denoised_image, noise_component

"""