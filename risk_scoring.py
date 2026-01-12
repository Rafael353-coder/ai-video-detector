def compute_risk_score(features, mode="normal"):
    """
    Calcula score de risco (0–100) com regras explicáveis
    e calibradas para reduzir falsos positivos.
    """

    # -------------------------
    # Baseline
    # -------------------------
    risk = 10 if mode == "normal" else 20
    reasons = []

    fv = features["face_variance"]
    fe = features["face_entropy"]
    ft = features["face_temporal"]
    gt = features["global_temporal"]

    # -------------------------
    # Regra 1 — Rosto ausente
    # -------------------------
    if fv == 0 and fe == 0:
        risk += 50
        reasons.append("Rosto humano não detetado")
        return min(risk, 100), reasons

    # -------------------------
    # Regra 2 — Movimento facial
    # -------------------------
    if ft == 0:
        risk += 30
        reasons.append("Movimento facial inexistente")
    elif ft < 5:
        risk += 10
        reasons.append("Movimento facial muito baixo")
    elif ft > 80:
        risk -= 10
        reasons.append("Movimento facial natural")

    # -------------------------
    # Regra 3 — Complexidade facial
    # -------------------------
    if fe < 4.8:
        risk += 25
        reasons.append("Complexidade facial muito baixa")
    elif fe < 5.5:
        risk += 10
        reasons.append("Complexidade facial reduzida")
    elif fe > 6.8:
        risk -= 10
        reasons.append("Alta complexidade facial")

    # -------------------------
    # Regra 4 — Movimento global (moderado!)
    # -------------------------
    if gt < 3:
        risk += 15
        reasons.append("Movimento global extremamente baixo")
    elif gt < 10:
        risk += 5
        reasons.append("Movimento global reduzido")
    elif gt > 40:
        risk -= 10
        reasons.append("Movimento global natural")

    # -------------------------
    # Regra 5 — Assinatura típica de IA (COMBINADA)
    # -------------------------
    if ft < 5 and fe < 5.2 and gt < 8:
        risk += 25
        reasons.append("Padrão facial típico de IA")

    # -------------------------
    # Modo strict
    # -------------------------
    if mode == "strict":
        risk += 10
        reasons.append("Modo de análise rigoroso")

    # -------------------------
    # Clamps
    # -------------------------
    risk = max(0, min(int(risk), 100))

    return risk, reasons

