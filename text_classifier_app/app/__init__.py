from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Necesario para requests y sessions
    app.config['SECRET_KEY'] = 'tu_clave_secreta'
    
    # Registrar blueprints
    from app.api.endpoints import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Registrar el blueprint de la interfaz web
    from app.web.routes import web_bp
    app.register_blueprint(web_bp)
    
    return app