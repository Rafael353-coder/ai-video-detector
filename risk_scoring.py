def compute_risk_score(features: dict, mode: str = "normal"):
    risk = 0
    reasons = []

    fv = features["face_variance"]
    fe = features["face_entropy"]
    ft = features["face_temporal"]
    gt = features["global_temporal"]

    # 🔹 Regras faciais
    if ft == 0:
        risk += 25
        reasons.append("Movimento facial inexistente")

    if fe < 5.5:
        risk += 20
        reasons.append("Complexidade facial baixa")

    if fv < 2000:
        risk += 15
        reasons.append("Textura facial artificial")

    # 🔹 Regras globais
    if gt < 15:
        risk += 20
        reasons.append("Movimento global demasiado estável")

    if gt > 120:
        risk += 10
        reasons.append("Movimento global errático")

    # 🔹 Modo estrito
    if mode == "strict":
        risk = int(risk * 1.2)

    risk = min(100, risk)

    if risk < 30:
        level = "BAIXO"
    elif risk < 60:
        level = "MÉDIO"
    else:
        level = "ALTO"

    return risk, level, reasons

