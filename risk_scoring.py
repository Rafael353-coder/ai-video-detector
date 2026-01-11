def compute_risk_score(features, mode="normal"):
    """
    Score de risco (0–100) com dois modos:
    - normal  → uso geral
    - strict  → verificação (mais agressivo)
    """

    # Baseline
    risk = 25 if mode == "strict" else 15
    reasons = []

    fv = features["face_variance"]
    fe = features["face_entropy"]
    ft = features["face_temporal"]
    gt = features["global_temporal"]

    # -------------------------
    # Rosto não detetado
    # -------------------------
    if fv == 0 and fe == 0 and ft == 0:
        risk += 70 if mode == "strict" else 55
        reasons.append("Rosto humano não detetado")

    else:
        # -------------------------
        # Movimento facial
        # -------------------------
        if ft < 20:
            risk += 40 if mode == "strict" else 25
            reasons.append("Movimento facial muito baixo")
        elif ft < 60:
            risk += 25 if mode == "strict" else 15
            reasons.append("Movimento facial limitado")
        else:
            risk += 20 if mode == "strict" else 5
            reasons.append("Movimento facial natural")

        # -------------------------
        # Complexidade facial
        # -------------------------
        if fe < 4:
            risk += 40 if mode == "strict" else 25
            reasons.append("Baixa complexidade facial")
        elif fe < 6:
            risk += 25 if mode == "strict" else 15
            reasons.append("Complexidade facial moderada")
        else:
            risk += 20 if mode == "strict" else 5
            reasons.append("Alta complexidade facial")

    # -------------------------
    # Movimento global
    # -------------------------
    if gt < 15:
        risk += 20 if mode == "strict" else 10
        reasons.append("Movimento global artificialmente suave")

    return min(100, max(0, risk)), reasons

