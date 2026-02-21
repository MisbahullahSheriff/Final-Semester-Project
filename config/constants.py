CONDITIONS = [
    "Common Cold",
    "Flu",
    "COVID-19",
    "Gastroenteritis",
    "Migraine",
    "Urinary Tract Infection",
    "Allergy",
    "Bronchitis"
]

TRIAGE_CLASSES = ["self-care", "doctor-visit", "emergency"]

SAMPLE_SYMPTOMS = {
    "Common Cold": [
        "runny nose and sneezing",
        "mild sore throat and congestion",
        "low fever and nasal congestion"
    ],
    "Flu": [
        "high fever body ache chills",
        "severe fatigue fever and cough",
        "sudden fever muscle pain headache"
    ],
    "COVID-19": [
        "loss of smell and persistent dry cough",
        "fever sore throat and breathlessness",
        "fatigue and loss of taste shortness of breath"
    ],
    "Gastroenteritis": [
        "diarrhea vomiting abdominal cramps",
        "stomach pain nausea and vomiting"
    ],
    "Migraine": [
        "severe one-sided headache with nausea",
        "throbbing headache light sensitivity"
    ],
    "Urinary Tract Infection": [
        "painful urination frequent urge lower belly pain",
        "burning while peeing cloudy urine and fever"
    ],
    "Allergy": [
        "itchy eyes runny nose watery eyes",
        "sneezing and itchy throat after pollen exposure"
    ],
    "Bronchitis": [
        "persistent productive cough chest discomfort",
        "cough producing mucus and wheeze"
    ]
}

CONDITION_TO_TRIAGE = {
    "Common Cold": "self-care",
    "Flu": "doctor-visit",
    "COVID-19": "doctor-visit",
    "Gastroenteritis": "doctor-visit",
    "Migraine": "doctor-visit",
    "Urinary Tract Infection": "doctor-visit",
    "Allergy": "self-care",
    "Bronchitis": "doctor-visit"
}

EMERGENCY_SYMPTOMS = [
    "severe chest pain radiating to arm",
    "sudden severe shortness of breath",
    "loss of consciousness",
    "sudden weakness on one side of body"
]