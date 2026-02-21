import os
from restaurant_ai_platform import create_app

app = create_app()

if name == “main”:
port = int(os.environ.get(“PORT”, 5000))
app.run(host=“0.0.0.0”, port=port)