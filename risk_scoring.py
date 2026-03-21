def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def compute_risk_score(features: dict, mode="normal"):
    score = 20
    reasons = []

    # -------- IMAGEM --------
    variance = features.get("variance")
    entropy = features.get("entropy")
    laplacian = features.get("laplacian")
    fft_energy = features.get("fft_energy")
    noise_level = features.get("noise_level")
    high_freq_ratio = features.get("high_freq_ratio")
    color_consistency = features.get("color_consistency")
    texture_score = features.get("texture_score")

    # -------- VIDEO --------
    face_variance = features.get("face_variance")
    face_entropy = features.get("face_entropy")
    face_temporal = features.get("face_temporal")
    global_temporal = features.get("global_temporal")

    # =========================
    # REGRAS PARA IMAGEM
    # =========================
    if variance is not None:
        if variance < 1200:
            score += 12
            reasons.append("Baixa variancia global")
        elif variance < 2200:
            score += 6

    if entropy is not None:
        if entropy < 5.2:
            score += 14
            reasons.append("Baixa complexidade visual")
        elif entropy < 6.0:
            score += 6

    if laplacian is not None:
        if laplacian < 80:
            score += 14
            reasons.append("Nitidez e detalhe anormalmente baixos")
        elif laplacian < 150:
            score += 6

    if noise_level is not None:
        if noise_level < 4:
            score += 14
            reasons.append("Ruido residual demasiado limpo")
        elif noise_level < 7:
            score += 6

    if high_freq_ratio is not None:
        if high_freq_ratio < 0.82:
            score += 10
            reasons.append("Pouca energia de alta frequencia")
        elif high_freq_ratio < 0.88:
            score += 4

    if color_consistency is not None:
        if color_consistency > 0.995:
            score += 8
            reasons.append("Canais de cor demasiado consistentes")

    if texture_score is not None:
        if texture_score < 12:
            score += 12
            reasons.append("Textura excessivamente suave")
        elif texture_score < 18:
            score += 5

    if fft_energy is not None:
        if fft_energy < 3.2:
            score += 8
            reasons.append("Estrutura frequencial pouco natural")
        elif fft_energy > 6.8:
            score += 5

    # =========================
    # REGRAS PARA VIDEO
    # =========================
    if face_entropy is not None and face_entropy < 5.3:
        score += 12
        reasons.append("Complexidade facial baixa")

    if face_temporal is not None and face_temporal < 3:
        score += 15
        reasons.append("Movimento facial demasiado estavel")

    if global_temporal is not None and global_temporal < 8:
        score += 10
        reasons.append("Movimento global artificial")

    if face_variance is not None and face_variance < 2000:
        score += 8
        reasons.append("Padrao facial demasiado regular")

    # =========================
    # AJUSTES ANTI-FALSO-POSITIVO
    # =========================
    positive_human_signals = 0

    if entropy is not None and entropy > 6.2:
        positive_human_signals += 1
    if noise_level is not None and noise_level > 8:
        positive_human_signals += 1
    if texture_score is not None and texture_score > 20:
        positive_human_signals += 1
    if global_temporal is not None and global_temporal > 15:
        positive_human_signals += 1
    if face_temporal is not None and face_temporal > 8:
        positive_human_signals += 1

    if positive_human_signals >= 2:
        score -= 12
    if positive_human_signals >= 3:
        score -= 8

    if mode == "strict":
        score += 8

    score = clamp(int(score), 0, 100)

    if score < 30:
        level = "BAIXO"
    elif score < 60:
        level = "MEDIO"
    elif score < 80:
        level = "SUSPEITO"
    else:
        level = "ALTO"

    if not reasons:
        reasons.append("Sem sinais fortes de artificio")

    return {
        "risk": score,
        "level": level,
        "reasons": reasons
    }
