import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename

from services.car_identifier import CarIdentificationError, identify_car_from_image

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            return handle_upload()

        return render_template(
            "index.html",
            result=None,
            error_message=None,
            recent_results=get_recent_results(),
            max_file_size_mb=MAX_CONTENT_LENGTH // (1024 * 1024),
        )

    @app.errorhandler(413)
    def file_too_large(_error):
        return render_template(
            "index.html",
            result=None,
            error_message="That image is too large. Please upload a file under 8 MB.",
            recent_results=get_recent_results(),
            max_file_size_mb=MAX_CONTENT_LENGTH // (1024 * 1024),
        ), 413

    def handle_upload():
        uploaded_file = request.files.get("car_image")

        if not uploaded_file or uploaded_file.filename == "":
            return render_with_error("Choose an image to analyze.")

        if not allowed_file(uploaded_file.filename):
            return render_with_error("Invalid file type. Please upload PNG, JPG, JPEG, or WEBP.")

        filename = generate_filename(uploaded_file.filename)
        saved_path = app.config["UPLOAD_FOLDER"] / filename

        try:
            uploaded_file.save(saved_path)
        except OSError:
            return render_with_error("Upload failure. Please try again with a different image.")

        try:
            result = identify_car_from_image(saved_path)
        except CarIdentificationError as error:
            return render_with_error(str(error), image_url=f"uploads/{filename}")

        if not result.get("make") and not result.get("model"):
            return render_with_error("No prediction found. Try a clearer photo of the car.", image_url=f"uploads/{filename}")

        result["image_url"] = f"uploads/{filename}"
        add_to_history(result)

        return render_template(
            "index.html",
            result=result,
            error_message=None,
            recent_results=get_recent_results(),
            max_file_size_mb=MAX_CONTENT_LENGTH // (1024 * 1024),
        )

    def render_with_error(message, image_url=None):
        return render_template(
            "index.html",
            result={"image_url": image_url} if image_url else None,
            error_message=message,
            recent_results=get_recent_results(),
            max_file_size_mb=MAX_CONTENT_LENGTH // (1024 * 1024),
        ), 400

    return app


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_filename(original_name):
    safe_name = secure_filename(original_name)
    suffix = Path(safe_name).suffix.lower()
    return f"{uuid.uuid4().hex}{suffix}"


def add_to_history(result):
    history = session.get("recent_results", [])
    history.insert(
        0,
        {
            "make": result.get("make") or "Unknown",
            "model": result.get("model") or "",
            "generation_or_trim": result.get("generation_or_trim"),
            "year_range": result.get("year_range"),
            "body_style": result.get("body_style"),
            "confidence": result.get("confidence"),
        },
    )
    session["recent_results"] = history[:5]
    session.modified = True


def get_recent_results():
    return session.get("recent_results", [])


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
