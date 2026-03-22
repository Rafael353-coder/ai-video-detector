import cv2
import numpy as np


def _safe_float(x, default=0.0):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


def _compute_image_like_features_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    variance = _safe_float(np.var(gray))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    entropy = _safe_float(-np.sum(hist * np.log2(hist + 1e-8)))

    laplacian = _safe_float(cv2.Laplacian(gray, cv2.CV_64F).var())

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    h, w = magnitude.shape
    center = magnitude[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    high_freq = _safe_float(np.mean(center))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float32) - blur.astype(np.float32)
    noise_score = _safe_float(np.var(noise))

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = _safe_float(np.mean(np.sqrt(np.maximum(sobelx*2 + sobely*2, 0))))

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    ok, encimg = cv2.imencode(".jpg", frame, encode_param)
    if ok:
        decimg = cv2.imdecode(encimg, 1)
        if decimg is not None:
            jpeg_artifacts = _safe_float(cv2.absdiff(frame, decimg).mean())
        else:
            jpeg_artifacts = 0.0
    else:
        jpeg_artifacts = 0.0

    blur_img = cv2.GaussianBlur(frame, (9, 9), 0)
    smoothness = _safe_float(cv2.absdiff(frame, blur_img).mean())

    local_contrast = _safe_float(gray.std())

    return {
        "variance": variance,
        "entropy": entropy,
        "laplacian": laplacian,
        "high_freq": high_freq,
        "noise": noise_score,
        "edge_strength": edge_strength,
        "jpeg_artifacts": jpeg_artifacts,
        "smoothness": smoothness,
        "local_contrast": local_contrast,
    }


def _frame_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    variance = _safe_float(np.var(gray))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    entropy = _safe_float(-np.sum(hist * np.log2(hist + 1e-8)))

    laplacian = _safe_float(cv2.Laplacian(gray, cv2.CV_64F).var())

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    fft_energy = _safe_float(np.mean(np.log1p(magnitude)))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = gray.astype(np.float32) - blur.astype(np.float32)
    noise_level = _safe_float(np.std(noise_residual))

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = max(1, min(rows, cols) // 8)

    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    total_energy = np.sum(magnitude) + 1e-8
    high_freq_energy = np.sum(magnitude * mask)
    high_freq_ratio = _safe_float(high_freq_energy / total_energy)

    b, g, r = cv2.split(frame)

    def safe_corr(a, b):
        try:
            if a.std() <= 1e-8 or b.std() <= 1e-8:
                return 0.0
            corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
            return _safe_float(corr, 0.0)
        except Exception:
            return 0.0

    rg_corr = safe_corr(r, g)
    rb_corr = safe_corr(r, b)
    gb_corr = safe_corr(g, b)
    color_consistency = _safe_float((rg_corr + rb_corr + gb_corr) / 3.0)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    texture_score = _safe_float(np.mean(np.sqrt(np.maximum(sobelx*2 + sobely*2, 0))))

    image_like = _compute_image_like_features_from_frame(frame)

    return {
        "variance_mean_base": variance,
        "entropy_mean_base": entropy,
        "laplacian_mean_base": laplacian,
        "fft_energy_mean_base": fft_energy,
        "noise_level_mean_base": noise_level,
        "high_freq_ratio_mean_base": high_freq_ratio,
        "color_consistency_mean_base": color_consistency,
        "texture_score_mean_base": texture_score,
        "jpeg_artifacts_mean_base": image_like["jpeg_artifacts"],
        "smoothness_mean_base": image_like["smoothness"],
        "local_contrast_mean_base": image_like["local_contrast"],
    }


def compute_features(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Nao foi possivel abrir o video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError("Video invalido ou sem frames.")

    sample_frames = min(40, frame_count)
    indices = np.linspace(0, frame_count - 1, sample_frames).astype(int)

    frame_feature_list = []
    temporal_diffs = []
    prev_gray = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        frame_feature_list.append(_frame_features(frame))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            temporal_diffs.append(_safe_float(np.mean(diff)))
        prev_gray = gray

    cap.release()

    if not frame_feature_list:
        raise ValueError("Nao foi possivel extrair frames do video.")

    keys = frame_feature_list[0].keys()
    aggregated = {}

    for key in keys:
        values = [_safe_float(f[key]) for f in frame_feature_list]
        base_name = key.replace("_mean_base", "")
        aggregated[f"{base_name}_mean"] = _safe_float(np.mean(values))
        aggregated[f"{base_name}_std"] = _safe_float(np.std(values))

    if temporal_diffs:
        aggregated["temporal_diff_mean"] = _safe_float(np.mean(temporal_diffs))
        aggregated["temporal_diff_std"] = _safe_float(np.std(temporal_diffs))
    else:
        aggregated["temporal_diff_mean"] = 0.0
        aggregated["temporal_diff_std"] = 0.0

    return aggregated


def extract_video_frames_for_image_model(video_path: str, max_frames: int = 10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Nao foi possivel abrir o video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError("Video invalido ou sem frames.")

    sample_frames = min(max_frames, frame_count)
    indices = np.linspace(0, frame_count - 1, sample_frames).astype(int)

    features_list = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        feats = _compute_image_like_features_from_frame(frame)
        features_list.append(feats)

    cap.release()

    return features_list
