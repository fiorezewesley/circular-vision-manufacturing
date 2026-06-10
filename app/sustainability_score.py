CIRCULARITY_RULES = {
    "battery": {
        "material_stream": "special hazardous disposal",
        "recommended_action": "send to certified reverse logistics or hazardous waste handling",
        "priority": "high",
        "circularity_score": 40,
        "estimated_landfill_avoidance_kg": 0.08,
        "special_handling": True,
    },
    "biological": {
        "material_stream": "organic waste stream",
        "recommended_action": "send to composting, biodigestion or controlled organic waste treatment",
        "priority": "medium",
        "circularity_score": 75,
        "estimated_landfill_avoidance_kg": 0.20,
        "special_handling": False,
    },
    "brown-glass": {
        "material_stream": "glass recycling stream",
        "recommended_action": "send to glass recycling stream after contamination check",
        "priority": "low",
        "circularity_score": 85,
        "estimated_landfill_avoidance_kg": 0.35,
        "special_handling": False,
    },
    "green-glass": {
        "material_stream": "glass recycling stream",
        "recommended_action": "send to glass recycling stream after contamination check",
        "priority": "low",
        "circularity_score": 85,
        "estimated_landfill_avoidance_kg": 0.35,
        "special_handling": False,
    },
    "white-glass": {
        "material_stream": "glass recycling stream",
        "recommended_action": "send to glass recycling stream after contamination check",
        "priority": "low",
        "circularity_score": 85,
        "estimated_landfill_avoidance_kg": 0.35,
        "special_handling": False,
    },
    "cardboard": {
        "material_stream": "recyclable packaging stream",
        "recommended_action": "send to cardboard and paper recycling stream",
        "priority": "low",
        "circularity_score": 90,
        "estimated_landfill_avoidance_kg": 0.12,
        "special_handling": False,
    },
    "paper": {
        "material_stream": "recyclable paper stream",
        "recommended_action": "send to paper recycling stream",
        "priority": "low",
        "circularity_score": 88,
        "estimated_landfill_avoidance_kg": 0.05,
        "special_handling": False,
    },
    "plastic": {
        "material_stream": "plastic recycling stream",
        "recommended_action": "send to plastic recycling stream after resin and contamination check",
        "priority": "medium",
        "circularity_score": 78,
        "estimated_landfill_avoidance_kg": 0.06,
        "special_handling": False,
    },
    "metal": {
        "material_stream": "metal recycling stream",
        "recommended_action": "send to metal recycling stream",
        "priority": "low",
        "circularity_score": 92,
        "estimated_landfill_avoidance_kg": 0.15,
        "special_handling": False,
    },
    "clothes": {
        "material_stream": "textile recovery stream",
        "recommended_action": "send to textile reuse, recovery or recycling flow",
        "priority": "medium",
        "circularity_score": 70,
        "estimated_landfill_avoidance_kg": 0.30,
        "special_handling": False,
    },
    "shoes": {
        "material_stream": "mixed material recovery stream",
        "recommended_action": "send to reuse, material recovery or specialized recycling flow",
        "priority": "medium",
        "circularity_score": 65,
        "estimated_landfill_avoidance_kg": 0.45,
        "special_handling": False,
    },
    "trash": {
        "material_stream": "non-recyclable waste stream",
        "recommended_action": "send to controlled disposal and review source reduction opportunities",
        "priority": "high",
        "circularity_score": 20,
        "estimated_landfill_avoidance_kg": 0.00,
        "special_handling": False,
    },
}


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", "-")


def build_sustainability_decision(predicted_class: str) -> dict:
    label = normalize_label(predicted_class)

    default_rule = {
        "material_stream": "unknown material stream",
        "recommended_action": "send to manual inspection before disposal decision",
        "priority": "high",
        "circularity_score": 30,
        "estimated_landfill_avoidance_kg": 0.00,
        "special_handling": True,
    }

    rule = CIRCULARITY_RULES.get(label, default_rule).copy()
    rule["predicted_class"] = label

    return rule


def get_sustainability_decision(predicted_class: str) -> dict:
    return build_sustainability_decision(predicted_class)


def calculate_sustainability_score(predicted_class: str) -> dict:
    return build_sustainability_decision(predicted_class)