from flask import Flask, request
from src.generate import generate


app = Flask(__name__)


@app.route("/generate", methods=["GET"])
def generate_handler():
    prompt = request.args.get("prompt")

    if not prompt:
        return "Missing prompt", 400

    return {
        "prompt": prompt,
        "image": generate(prompt),
    }


def main():
    app.run()


if __name__ == "__main__":
    main()
