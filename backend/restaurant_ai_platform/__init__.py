from flask import Flask, jsonify

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return jsonify({"message": "Restaurant AI Platform Backend is running"})

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app