from typing import Union, Dict

import PIL
import torch
from fastapi import APIRouter, File, HTTPException, Request
from starlette.status import HTTP_400_BAD_REQUEST

from app.api import models as api_models
from app.api.utils import evaluation as evaluation_utils
from app.api.utils import images as image_utils

router = APIRouter()


def classify(image: PIL.Image, model: torch.nn.Module) -> Dict[str, Union[str, float]]:
    """
    Classifies the input `image` using the specified `model`.

    Arguments:
        image {PIL.Image} -- Input image to be classified.
        model {torch.nn.Module} -- PyTorch model used to classify the image.

    Returns:
        Dict[str, Union[str, float]] -- Dictionary with the classification result.
    """
    image = image_utils.preprocess_image(image)
    with torch.no_grad():
        logits = model(image.unsqueeze(0))

    return evaluation_utils.to_result_dict(logits)


@router.post("/file", response_model=api_models.ClassificationResult)
async def classify_file(request: Request, image: bytes = File(...)):
    """
    Clasifies an image as dog or cat. The input image is received as a stream of bytes.

    Arguments:
        request {Request} -- The user request.

    Keyword Arguments:
        image {bytes} -- Input image (as bytes) to be classified.

    Raises:
        HTTPException: When the input image could not be loaded.

    Returns:
        models.ClassificationResult -- JSON object with the classification result.
    """
    try:
        image = image_utils.bytes_to_image(image)
    except Exception:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, detail="Error loading input image",
        )
    return classify(image, request.app.state.model)


@router.post("/base64", response_model=api_models.ClassificationResult)
async def classify_base64(request: Request, data: api_models.InputImage):
    """
    Clasifies an image as dog or cat. The input image is encoded in base64 format.

    Arguments:
        request {Request} -- The user request.

    Keyword Arguments:
        image {bytes} -- Input image (as bytes) to be classified.

    Raises:
        HTTPException: When the input image could not be loaded.

    Returns:
        models.ClassificationResult-- JSON object with the classification result.
    """
    image = data.image
    try:
        image = image_utils.base64_to_image(image)
    except Exception:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, detail="Error decoding input image",
        )
    return classify(image, request.app.state.model)
