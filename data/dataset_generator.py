import random
import pandas as pd
from config.config import EMERGENCY_RATE
from config.constants import (
    CONDITIONS,
    SAMPLE_SYMPTOMS,
    CONDITION_TO_TRIAGE,
    EMERGENCY_SYMPTOMS
)


def generate_dataset(n_samples=12000):
    rows = []

    for _ in range(n_samples):
        condition = random.choice(CONDITIONS)
        symptom = random.choice(SAMPLE_SYMPTOMS[condition])
        triage = CONDITION_TO_TRIAGE[condition]

        if random.random() < EMERGENCY_RATE:
            symptom += ", " + random.choice(EMERGENCY_SYMPTOMS)
            triage = "emergency"

        rows.append({
            "symptoms_text": symptom,
            "triage_class": triage
        })

    return pd.DataFrame(rows)