from app import create_app
from config import DEBUG, PORT

app = create_app()

if __name__ == '__main__':
    app.run(debug=DEBUG, port=PORT)