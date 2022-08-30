from time import time

from app.main import backtest, realtime, training
from config import Config
from flask import Flask


def create_app(config_class=Config):
    start = time()
    app = Flask(__name__)
    app.config.from_object(config_class)

    print(f"\033[1;31mTime to initiate App {time()-start:.4f}s \033[1;37m")


    if app.config['MODE'] == 'backtest':
        backtest.process_backtest(app.config,start)
    elif app.config['MODE'] == 'training':
        training.process_training(app.config, start)
    else:
        realtime.process_realtime(app.config, start)


    # Blueprint Api
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
