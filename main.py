from flask import Flask, request
from time import sleep


app = Flask(__name__)


@app.route("/generate", methods=["GET"])
def generate_handler():
    prompt = request.args.get("prompt")

    if not prompt:
        return "Missing prompt", 400

    # sleep(2)

    return {
        "prompt": prompt,
        "generated": "This is a generated response",
    }


def main():
    app.run()
    pass


if __name__ == "__main__":
    main()
