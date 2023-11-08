from flask import Flask, request
from src.generate import generate


app = Flask(__name__)


@app.route("/generate", methods=["GET"])
def generate_handler():
    prompt = request.args.get("prompt")
    height = request.args.get("h")
    width = request.args.get("w")
    steps = request.args.get("s")

    if height == None:
        height = 512

    if width == None:
        width = 512

    if steps == None:
        steps = 50

    if not prompt:
        return "Missing prompt", 400

    return {
        "prompt": prompt,
        "image": generate(prompt, int(height), int(width), int(steps)),
    }


@app.route("/system-status", methods=['GET'])
def get_status():
    return {"msg": "System is running"}, 200


def main():
    app.run()


if __name__ == "__main__":
    main()
