"""FastAPI models."""
from pydantic import BaseModel  # pylint: disable=no-name-in-module


class ClassificationResult(BaseModel):
    """Cat vs Dog classification result"""

    label: str
    confidence: float


class InputImage(BaseModel):
    """Input image encoded in base64 format"""

    image: str
