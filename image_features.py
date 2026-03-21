import cv2
import numpy as np
from scipy.stats import entropy


def compute_image_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Nao foi possivel abrir imagem")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variance = float(np.var(gray))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-8)
    ent = float(entropy(hist.flatten()))

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = float(lap.var())

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    center = magnitude[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    high_freq = float(np.mean(center))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float32) - blur.astype(np.float32)
    noise_score = float(np.var(noise))

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = float(np.mean(np.sqrt(np.maximum(sobelx*2 + sobely*2, 0))))

    return {
        "variance": variance,
        "entropy": ent,
        "laplacian": laplacian,
        "high_freq": high_freq,
        "noise": noise_score,
        "edge_strength": edge_strength
    }
