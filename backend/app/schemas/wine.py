from pydantic import BaseModel, Field
from typing import Literal

class WineInput(BaseModel):
    fixed_acidity: float = Field(..., ge=0, example=7.4, description="Fixed acidity (g/dm³)")
    volatile_acidity: float = Field(..., ge=0, example=0.7, description="Volatile acidity (g/dm³)")
    citric_acid: float = Field(..., ge=0, example=0.0, description="Citric acid (g/dm³)")
    residual_sugar: float = Field(..., ge=0, example=1.9, description="Residual sugar (g/dm³)")
    chlorides: float = Field(..., ge=0, example=0.076, description="Chlorides (g/dm³)")
    free_sulfur_dioxide: float = Field(..., ge=0, example=11.0, description="Free sulfur dioxide (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., ge=0, example=34.0, description="Total sulfur dioxide (mg/dm³)")
    density: float = Field(..., ge=0, example=0.9978, description="Density (g/cm³)")
    pH: float = Field(..., ge=0, le=14, example=3.51, description="pH value (0–14)")
    sulphates: float = Field(..., ge=0, example=0.56, description="Sulphates (g/dm³)")
    alcohol: float = Field(..., ge=0, example=9.4, description="Alcohol (% by volume)")
    type: Literal["red", "white"] = Field(example="red", description="Wine type", default="red")

