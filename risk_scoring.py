def compute_risk_score(features, mode="normal"):
    """
    Calcula score de risco (0–100).
    Retorna sempre: risk, level, reasons
    """

    reasons = []

    # -------------------------
    # Baseline
    # -------------------------
    risk = 15 if mode == "normal" else 25

    fv = features.get("face_variance", 0.0)
    fe = features.get("face_entropy", 0.0)
    ft = features.get("face_temporal", 0.0)
    gt = features.get("global_temporal", 0.0)

    # -------------------------
    # Regra 1 — Rosto não detetado
    # -------------------------
    if fv == 0 and fe == 0:
        risk += 50
        reasons.append("Rosto humano não detetado")

    # -------------------------
    # Regra 2 — Movimento facial
    # -------------------------
    if ft < 5:
        risk += 30
        reasons.append("Movimento facial inexistente")
    elif ft < 15:
        risk += 20
        reasons.append("Movimento facial artificial")
    elif ft < 30:
        risk += 10
        reasons.append("Movimento facial pouco natural")
    else:
        reasons.append("Movimento facial natural")

    # -------------------------
    # Regra 3 — Complexidade facial
    # -------------------------
    if fe < 6.0:
        risk += 25
        reasons.append("Complexidade facial muito baixa")
    elif fe < 6.8:
        risk += 15
        reasons.append("Complexidade facial suspeita")
    else:
        reasons.append("Complexidade facial natural")

    # -------------------------
    # Regra 4 — Movimento global
    # -------------------------
    if gt < 20:
        risk += 15
        reasons.append("Movimento global artificial")
    elif gt > 130:
        risk += 10
        reasons.append("Movimento global excessivo")

    # -------------------------
    # 🔥 REGRAS DE COMBINAÇÃO (BOOST)
    # -------------------------
    if ft < 15 and fe < 6.8:
        risk += 15
        reasons.append("Padrão facial típico de IA")

    if ft < 10 and gt < 25:
        risk += 15
        reasons.append("Movimento demasiado estável para humano")

    # -------------------------
    # Normalização
    # -------------------------
    risk = max(0, min(100, int(risk)))

    if risk < 30:
        level = "BAIXO"
    elif risk < 60:
        level = "MÉDIO"
    else:
        level = "ALTO"

    return risk, level, reasons

