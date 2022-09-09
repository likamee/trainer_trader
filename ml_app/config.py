import os

from dotenv import load_dotenv
from oandapyV20 import API as Con

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

    if os.environ.get('ENV') == 'live':
        ACCID = os.environ.get('REAL_OANDA_ACCOUNT')
        TOKEN = os.environ.get('REAL_OANDA_TOKEN')
    else:
        ACCID = os.environ.get('DEMO_OANDA_ACCOUNT')
        TOKEN = os.environ.get('DEMO_OANDA_TOKEN')

    API = Con(access_token=TOKEN, environment=os.environ.get('ENV'))
    MODE = os.environ.get('MODE')
    UPDATE_DATA = os.environ.get('UPDATE_DATA')
    STRATEGIES = ['ASH', 'SPEAR']
    NCANDLEST = 336*5  # multiplo de 48
    NCANDLESV = 96  # 80
    NCANDLESSAFESTART = 288
    NCANDLESSAFET = 48  # 48
    MINICIAL = 10000
    EARNINGS = [MINICIAL, MINICIAL, MINICIAL, MINICIAL]
    LIM_WR = 68
    NCANDLESR = [30, 50, 100, 150, 200]

    RISK = 0.007
    C_PREJATR = 1.5
    C_LUCATR = 1
    ATR_CUT = 90
    SPREAD_CUT = 90

    # variaveis de parametros
    PARAM_FIT = 7
    PARAMS_MIN = {'ASH': [5, 5, 5, 1], 'SPEAR': [5, 5, 1]}
    PARAMS_MAX = {'ASH': [60, 60, 60, 5], 'SPEAR': [60, 60, 5]}

    # variaveis do optimizaer
    ITERS = 50  # 50
    NRUNS = 2  # 2
    IMAXREPEATS = 15  # 10

    # PSO
    N_IND = [400, 150]  # ASH, SPEARMAN
    W = 0.5
    C1 = 1
    C2 = 1

    BARS = {'BTCUSD': 0}

    AUX_BARS = BARS.copy()

    PAIRS = list(BARS.keys())
