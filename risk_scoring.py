def compute_risk_score(features, mode="normal"):
    """
    Calcula um score de risco (0–100) e motivos explicáveis
    baseado em comportamento facial e movimento temporal.
    """

    # -------------------------
    # BASE
    # -------------------------
    risk = 0
    reasons = []

    fv = features.get("face_variance", 0.0)
    fe = features.get("face_entropy", 0.0)
    ft = features.get("face_temporal", 0.0)
    gt = features.get("global_temporal", 0.0)

    # -------------------------
    # PARÂMETROS POR MODO
    # -------------------------
    if mode == "strict":
        FACE_MISSING_PENALTY = 85
        LOW_ENTROPY_PENALTY = 30
        LOW_TEMPORAL_PENALTY = 25
    else:
        FACE_MISSING_PENALTY = 70
        LOW_ENTROPY_PENALTY = 20
        LOW_TEMPORAL_PENALTY = 15

    # =====================================================
    # REGRA 1 — ROSTO NÃO DETETADO (CRÍTICA)
    # =====================================================
    if fv == 0 and fe == 0:
        risk += FACE_MISSING_PENALTY
        reasons.append("Rosto humano não detetado")

    # =====================================================
    # REGRA 2 — ROSTO ESTÁTICO / ARTIFICIAL (MUITO FORTE)
    # =====================================================
    if ft < 5:
        risk += 45
        reasons.append("Rosto estático ou artificial")

    # =====================================================
    # REGRA 3 — MOVIMENTO FACIAL POUCO NATURAL
    # =====================================================
    elif ft < 40:
        risk += LOW_TEMPORAL_PENALTY
        reasons.append("Movimento facial pouco natural")

    # =====================================================
    # REGRA 4 — BAIXA COMPLEXIDADE FACIAL
    # =====================================================
    if fe < 6.5:
        risk += LOW_ENTROPY_PENALTY
        reasons.append("Baixa complexidade facial")

    # =====================================================
    # REGRA 5 — PADRÃO CLÁSSICO DE DEEPFAKE
    # (Rosto existe mas comporta-se mal)
    # =====================================================
    if fv > 0 and fe < 7.0 and ft < 10:
        risk += 25
        reasons.append("Rosto com comportamento não humano")

    # =====================================================
    # REGRA 6 — MOVIMENTO GLOBAL SEM EXPRESSÃO FACIAL
    # =====================================================
    if gt > 90 and ft < 10:
        risk += 20
        reasons.append("Movimento global inconsistente com rosto")

    # =====================================================
    # REGRA 7 — VÍDEO DEMASIADO LIMPO / ARTIFICIAL
    # =====================================================
    if fe < 6.8 and fv < 2500:
        risk += 15
        reasons.append("Textura facial artificial")

    # =====================================================
    # NORMALIZAÇÃO FINAL
    # =====================================================
    risk = int(min(100, max(0, risk)))

    # =====================================================
    # NÍVEL
    # =====================================================
    if risk >= 70:
        level = "ALTO"
    elif risk >= 40:
        level = "MÉDIO"
    else:
        level = "BAIXO"

    return risk, level, reasons

