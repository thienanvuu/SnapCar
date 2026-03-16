# SnapCar

A portfolio-ready Flask demo that lets users upload a car photo and get an AI-powered guess of the make and model using OpenAI vision.

## Features

- Clean, responsive upload experience with drag-and-drop support
- Polished result card with likely match, pricing context, year range, and body style
- Session-based history of recent analyses
- OpenAI Responses API integration for real image analysis
- Rough MSRP and current-value estimates plus a Wikipedia link
- A small list of 2-3 alternate likely matches for ambiguous photos
- Mock mode for demos without a live API key
- Isolated AI service layer for easy model/provider swaps later

## Project structure

```text
.
├── app.py
├── services/
│   ├── __init__.py
│   └── car_identifier.py
├── static/
│   ├── css/styles.css
│   ├── js/app.js
│   └── uploads/
├── templates/
│   ├── base.html
│   └── index.html
├── .env.example
├── requirements.txt
└── README.md
```

## Local setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the environment example and add your OpenAI API key:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env`:

   ```env
   OPENAI_API_KEY=your-real-openai-key
   AI_MODEL=gpt-4o-mini
   MOCK_AI_MODE=false
   ```

4. Start the app:

   ```bash
   flask --app app run --debug
   ```

5. Open the local URL shown in your terminal.

## Environment variables

- `FLASK_SECRET_KEY`: Session secret for Flask
- `OPENAI_API_KEY`: Your OpenAI API key for live image analysis
- `AI_MODEL`: The OpenAI model to use. The default is `gpt-4o-mini`
- `MOCK_AI_MODE`: Set to `false` for live OpenAI results, `true` for sample demo responses

## Where to add the AI integration

The main integration file is [services/car_identifier.py](/Users/thienanvu/Documents/SnapCar/services/car_identifier.py).

- `call_vision_model(...)` is already wired for OpenAI's Responses API
- Keep `identify_car_from_image(...)` as the app-facing entry point
- Change `AI_MODEL` in `.env` to swap OpenAI models later without touching the Flask routes
- Reuse `build_prompt()` so the model is consistently asked for your automotive JSON schema
- If you ever want a different provider later, replace only `call_vision_model(...)`

## OpenAI integration notes

This app sends:

- The uploaded image as a base64 data URL
- Your automotive identification prompt
- A strict JSON schema so the model returns the exact fields the UI expects


The default model is `gpt-4o-mini`, which OpenAI documents as supporting image input and structured outputs. If you want a stronger model later, change `AI_MODEL` in `.env` to another current OpenAI model that supports those features.

## Notes

- Uploaded files are stored temporarily in `static/uploads/` for the local demo
- The app validates file type and maximum size before analysis
- If you want production storage later, replace local file saving with object storage or in-memory processing
- If `MOCK_AI_MODE=true`, the app uses sample car guesses instead of calling OpenAI
