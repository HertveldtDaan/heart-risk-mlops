"""Request/response schemas.

The request model is GENERATED from the shared feature contract
(heartrisk.features), so the API accepts exactly the categories the model
was trained on. A typo like "Mael" is rejected with a 422 error.

All questions are REQUIRED. In the survey data a missing answer is
informative (people who skip questions are on average less healthy), so the
model learned "missing = higher risk". If the API allowed empty answers, an
app that simply forgot to ask a question would silently inflate the risk.
Only BMI may be replaced by height + weight.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from heartrisk.features import CATEGORICAL_FEATURES, PLAUSIBLE_RANGES

EXAMPLE = {
    "Sex": "Male",
    "AgeCategory": "Age 60 to 64",
    "GeneralHealth": "Fair",
    "LastCheckupTime": "Within past year (anytime less than 12 months ago)",
    "PhysicalActivities": "No",
    "RemovedTeeth": "6 or more, but not all",
    "HadStroke": "No",
    "HadAsthma": "No",
    "HadSkinCancer": "No",
    "HadCOPD": "Yes",
    "HadDepressiveDisorder": "No",
    "HadKidneyDisease": "No",
    "HadArthritis": "Yes",
    "HadDiabetes": "Yes",
    "DeafOrHardOfHearing": "No",
    "BlindOrVisionDifficulty": "No",
    "DifficultyConcentrating": "No",
    "DifficultyWalking": "Yes",
    "DifficultyDressingBathing": "No",
    "DifficultyErrands": "No",
    "SmokerStatus": "Former smoker",
    "ECigaretteUsage": "Never used e-cigarettes in my entire life",
    "AlcoholDrinkers": "No",
    "PhysicalHealthDays": 10,
    "MentalHealthDays": 2,
    "SleepHours": 6,
    "HeightInMeters": 1.78,
    "WeightInKilograms": 98,
}

class _PatientBase(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    @model_validator(mode="after")
    def bmi_or_height_and_weight(self):
        if self.BMI is None and (self.HeightInMeters is None or self.WeightInKilograms is None):
            raise ValueError("Give either BMI, or both HeightInMeters and WeightInKilograms")
        return self


_fields = {}
for name, categories in CATEGORICAL_FEATURES.items():
    _fields[name] = (Literal[tuple(categories)], Field(...))
for name in ("PhysicalHealthDays", "MentalHealthDays", "SleepHours"):
    low, high = PLAUSIBLE_RANGES[name]
    _fields[name] = (float, Field(..., ge=low, le=high))
_fields["BMI"] = (Optional[float], Field(default=None, ge=12, le=80,
                                          description="Optional when height and weight are given"))
_fields["HeightInMeters"] = (Optional[float], Field(default=None, ge=0.9, le=2.5))
_fields["WeightInKilograms"] = (Optional[float], Field(default=None, ge=20, le=300))

PatientInput = create_model("PatientInput", __base__=_PatientBase, **_fields)
PatientInput.__doc__ = "Answers of one person to the health questionnaire (all questions required)."


class Factor(BaseModel):
    feature: str
    value: Optional[str | float] = None
    impact: float = Field(description="SHAP value in log-odds: > 0 raises the risk, < 0 lowers it")
    direction: Literal["increases risk", "decreases risk"]


class ModelRef(BaseModel):
    name: str
    version: str


class PredictionResponse(BaseModel):
    id: int
    created_at: datetime
    risk_probability: float = Field(description="Estimated probability (0-1)")
    risk_level: Literal["low", "elevated", "high"]
    above_threshold: bool
    threshold: float
    top_factors: list[Factor]
    model: ModelRef
    disclaimer: str


class PredictionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    model_name: str
    model_version: str
    risk_probability: float
    risk_level: str
    above_threshold: bool
    input: dict
    top_factors: list
