from flask import Flask, jsonify

from . import api_serving


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return jsonify({"message": "Restaurant AI Platform Backend is running"})

    # Register API blueprint (health + pipeline endpoints)
    api_serving.register(app)

    return app