def retention_decision(prob, row, threshold=0.25):
    if prob >= threshold:
        if row["Contract"] == "Month-to-month":
            return "Offer retention discount"
        elif row["tenure"] < 6:
            return "Priority retention call"
        return "Retention email"
    return "No action"