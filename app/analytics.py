def ai_summary(avg, high, low):
    if avg > 0.7:
        return "⚠ Customer base is highly unstable. Immediate retention action required."
    elif avg > 0.4:
        return "⚡ Moderate churn risk detected. Monitor trends."
    else:
        return "✅ Customer portfolio stable."
