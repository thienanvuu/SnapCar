import base64
import json
import mimetypes
import os
import random
import re
from pathlib import Path
from urllib.parse import quote


class CarIdentificationError(Exception):
    """Raised when car identification fails in a user-facing way."""


def identify_car_from_image(image_path: Path) -> dict:
    """
    Identify the car in an image.

    This function is the main integration point for your AI vision model.
    It returns a normalized result that the UI can render safely.
    """
    use_mock_mode = os.getenv("MOCK_AI_MODE", "false").lower() == "true"

    if use_mock_mode:
        return mock_identification(image_path)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_API_KEY")
    model_name = os.getenv("AI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise CarIdentificationError(
            "AI response failure. Add OPENAI_API_KEY to your environment or enable MOCK_AI_MODE=true."
        )

    try:
        raw_response = call_vision_model(image_path=image_path, api_key=api_key, model_name=model_name)
    except CarIdentificationError:
        raise
    except Exception as error:
        raise CarIdentificationError(f"AI response failure: {error}") from error

    parsed = parse_ai_response(raw_response)

    if not parsed:
        raise CarIdentificationError("No prediction found. Try another image with a clearer view of the car.")

    return parsed


def call_vision_model(image_path: Path, api_key: str, model_name: str):
    """
    Send the image to the OpenAI Responses API and request strict JSON output.

    Official docs used for this integration:
    - Responses API supports image inputs and text outputs:
      https://developers.openai.com/api/reference/overview
    - Structured outputs in Responses use `text.format` with `json_schema`:
      https://developers.openai.com/api/docs/guides/migrate-to-responses
    """
    try:
        from openai import OpenAI
    except ImportError as error:
        raise CarIdentificationError(
            "AI response failure. Install dependencies from requirements.txt to enable OpenAI."
        ) from error

    client = OpenAI(api_key=api_key)

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    image_data_url = f"data:{mime_type};base64,{image_base64}"

    response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": build_prompt()},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "car_identification",
                    "strict": True,
                    "schema": response_schema(),
                }
            },
        )

    if getattr(response, "output_text", None):
        return response.output_text

    if hasattr(response, "model_dump"):
        return response.model_dump()

    return response


def build_prompt() -> str:
    return (
        "You are an expert automotive vision assistant.\n\n"
        "Identify the primary vehicle in this image.\n\n"
        "Rules:\n"
        "1. Use only visible evidence from the image.\n"
        "2. Do not hallucinate exact trims, years, or features if they are unclear.\n"
        "3. If the exact model is uncertain, provide the closest likely match.\n"
        "4. For pricing, provide only rough market estimates and use null if you cannot make a reasonable guess.\n"
        "5. Keep the reasoning short and grounded in visible cues such as headlights, grille shape, roofline, badge placement, proportions, or taillight design.\n"
        "6. Also provide 2 to 3 alternate likely candidates in `other_candidates` as simple strings like make and model only.\n"
        "7. Return only JSON, with no markdown and no extra commentary.\n\n"
        "Required JSON output:\n"
        "{\n"
        '  "make": "",\n'
        '  "model": "",\n'
        '  "generation_or_trim": "",\n'
        '  "year_range": "",\n'
        '  "body_style": "",\n'
        '  "confidence": "",\n'
        '  "estimated_market_value": "",\n'
        '  "msrp_when_new": "",\n'
        '  "best_guess_summary": "",\n'
        '  "reasoning": "",\n'
        '  "other_candidates": []\n'
        "}"
    )


def parse_ai_response(raw_response) -> dict:
    if isinstance(raw_response, dict):
        extracted = extract_output_text_from_response_dict(raw_response)
        if extracted:
            return parse_ai_response(extracted)
        return normalize_result(raw_response)

    if not raw_response:
        return {}

    if isinstance(raw_response, str):
        json_candidate = extract_json_block(raw_response)
        if json_candidate:
            try:
                return normalize_result(json.loads(json_candidate))
            except json.JSONDecodeError:
                pass

        return normalize_result(parse_unstructured_text(raw_response))

    return {}


def extract_output_text_from_response_dict(payload: dict):
    output_text = payload.get("output_text")
    if output_text:
        return output_text

    for item in payload.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return content["text"]

    return None


def extract_json_block(text: str):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def parse_unstructured_text(text: str) -> dict:
    lines = [line.strip("-* \n\t") for line in text.splitlines() if line.strip()]
    result = {
        "make": None,
        "model": None,
        "generation_or_trim": None,
        "year_range": None,
        "body_style": None,
        "confidence": None,
        "estimated_market_value": None,
        "msrp_when_new": None,
        "best_guess_summary": None,
        "reasoning": text.strip(),
        "other_candidates": [],
    }

    for line in lines:
        lower = line.lower()
        if lower.startswith("make:"):
            result["make"] = line.split(":", 1)[1].strip()
        elif lower.startswith("model:"):
            result["model"] = line.split(":", 1)[1].strip()
        elif lower.startswith("generation_or_trim:") or lower.startswith("generation/trim:"):
            result["generation_or_trim"] = line.split(":", 1)[1].strip()
        elif "year" in lower:
            result["year_range"] = line.split(":", 1)[-1].strip()
        elif "body" in lower or "style" in lower:
            result["body_style"] = line.split(":", 1)[-1].strip()
        elif lower.startswith("confidence:"):
            result["confidence"] = line.split(":", 1)[1].strip()
        elif lower.startswith("estimated_market_value:") or lower.startswith("estimated market value:"):
            result["estimated_market_value"] = line.split(":", 1)[1].strip()
        elif lower.startswith("msrp_when_new:") or lower.startswith("msrp when new:"):
            result["msrp_when_new"] = line.split(":", 1)[1].strip()
        elif lower.startswith("best_guess_summary:") or lower.startswith("summary:"):
            result["best_guess_summary"] = line.split(":", 1)[1].strip()
        elif lower.startswith("reasoning:"):
            result["reasoning"] = line.split(":", 1)[1].strip()

    if not result["make"] and lines:
        guess = lines[0]
        parts = guess.split()
        if len(parts) >= 2:
            result["make"] = parts[0]
            result["model"] = " ".join(parts[1:])

    return result


def normalize_result(data: dict) -> dict:
    best_guess_summary = clean_value(data.get("best_guess_summary")) or build_best_guess_summary(data)
    reasoning = clean_value(data.get("reasoning")) or "Visible exterior cues suggest this is the closest likely match."
    make = clean_value(data.get("make"))
    model = clean_value(data.get("model"))
    generation_or_trim = clean_value(data.get("generation_or_trim"))
    return {
        "make": make,
        "model": model,
        "generation_or_trim": generation_or_trim,
        "year_range": clean_value(data.get("year_range")),
        "body_style": clean_value(data.get("body_style")),
        "confidence": clean_value(data.get("confidence")) or "Likely match",
        "estimated_market_value": clean_value(data.get("estimated_market_value")),
        "msrp_when_new": clean_value(data.get("msrp_when_new")),
        "wikipedia_url": build_wikipedia_url(make, model, generation_or_trim),
        "best_guess_summary": best_guess_summary,
        "reasoning": reasoning,
        "other_candidates": normalize_candidates(data.get("other_candidates"), make, model),
        "short_explanation": best_guess_summary,
    }


def response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "make": {"type": "string"},
            "model": {"type": "string"},
            "generation_or_trim": {"type": "string"},
            "year_range": {"type": "string"},
            "body_style": {"type": "string"},
            "confidence": {"type": "string"},
            "estimated_market_value": {"type": "string"},
            "msrp_when_new": {"type": "string"},
            "best_guess_summary": {"type": "string"},
            "reasoning": {"type": "string"},
            "other_candidates": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "make",
            "model",
            "generation_or_trim",
            "year_range",
            "body_style",
            "confidence",
            "estimated_market_value",
            "msrp_when_new",
            "best_guess_summary",
            "reasoning",
            "other_candidates",
        ],
        "additionalProperties": False,
    }


def normalize_candidates(candidates, make, model):
    primary = " ".join(part for part in (make, model) if part).strip().lower()

    if not isinstance(candidates, list):
        return []

    cleaned = []
    seen = set()

    for candidate in candidates:
        value = clean_value(candidate)
        if not value:
            continue
        normalized = value.lower()
        if normalized == primary or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(value)

    return cleaned[:3]


def clean_value(value):
    if value is None:
        return None

    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in {"null", "unknown", "n/a", "na"}:
        return None
    return cleaned


def build_wikipedia_url(make, model, generation_or_trim):
    del generation_or_trim
    parts = [part for part in (make, model) if part]
    if not parts:
        return None
    query = " ".join(parts)
    return f"https://en.wikipedia.org/w/index.php?search={quote(query)}"


def build_best_guess_summary(data: dict):
    make = clean_value(data.get("make")) or "Unknown"
    model = clean_value(data.get("model"))
    confidence = clean_value(data.get("confidence")) or "Likely match"

    if model:
        return f"{confidence}: {make} {model}."

    return f"{confidence}: {make}."


def mock_identification(image_path: Path) -> dict:
    """
    Demo-friendly fallback so the app still works without a real API key.
    """
    samples = [
        {
            "make": "Porsche",
            "model": "911 Carrera",
            "generation_or_trim": "992 generation",
            "year_range": "2019-2024",
            "body_style": "Coupe",
            "confidence": "High confidence",
            "estimated_market_value": "$110,000-$145,000 used",
            "msrp_when_new": "$114,000-$130,000",
            "best_guess_summary": "High confidence: Porsche 911 Carrera, likely from the 992 generation.",
            "reasoning": "Rounded headlights, low coupe proportions, and the classic rear-engine 911 silhouette are visible.",
            "other_candidates": ["Porsche 718 Cayman", "Chevrolet Corvette Stingray", "Audi R8"],
        },
        {
            "make": "Tesla",
            "model": "Model 3",
            "generation_or_trim": None,
            "year_range": "2021-2024",
            "body_style": "Sedan",
            "confidence": "Likely match",
            "estimated_market_value": "$22,000-$35,000 used",
            "msrp_when_new": "$38,000-$52,000",
            "best_guess_summary": "Likely match: Tesla Model 3.",
            "reasoning": "Smooth nose, short hood, and compact fastback-like sedan proportions match a Model 3.",
            "other_candidates": ["Tesla Model S", "Hyundai Ioniq 6", "Polestar 2"],
        },
        {
            "make": "Toyota",
            "model": "RAV4",
            "generation_or_trim": "Fifth generation",
            "year_range": "2019-2024",
            "body_style": "SUV",
            "confidence": "Moderate confidence",
            "estimated_market_value": "$24,000-$38,000 used",
            "msrp_when_new": "$27,000-$40,000",
            "best_guess_summary": "Moderate confidence: Toyota RAV4, likely fifth generation.",
            "reasoning": "Upright crossover shape, squared wheel arches, and the angular front-end design point toward a RAV4.",
            "other_candidates": ["Honda CR-V", "Mazda CX-5", "Subaru Forester"],
        },
    ]

    random.seed(image_path.name)
    return normalize_result(random.choice(samples))
