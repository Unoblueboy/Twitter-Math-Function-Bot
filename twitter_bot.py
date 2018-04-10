'''This module is used to post to bot'''
import configparser
import numpy as np
import tweepy

from function_generator import FunctionGenerator as FG
import domain_colouring as d_c

config = configparser.ConfigParser()
config.read('config.ini')

consumer_key = config['Twitter']['consumer_key']
consumer_secret = config['Twitter']['consumer_secret']
access_token = config['Twitter']['access_token']
access_token_secret = config['Twitter']['access_token_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

unary_ops = {
    "sin": lambda x: np.sin(x),
    "cos": lambda x: np.cos(x),
    "tan": lambda x: np.tan(x),
    "sinh": lambda x: np.sinh(x),
    "cosh": lambda x: np.cosh(x),
    "tanh": lambda x: np.tanh(x),
    "repr": lambda x: 1 / complex(x),
    "exp": lambda x: np.exp(x),
    "log": lambda x: np.log(x),
}
binary_ops = {
    "add": lambda x, y: complex(x) + complex(y),
    "sub": lambda x, y: complex(x) - complex(y),
    "mul": lambda x, y: complex(x) * complex(y),
    "div": lambda x, y: complex(x) / complex(y),
}

while True:
    done = False
    try:
        f_gen = FG(unary_op=unary_ops, binary_op=binary_ops)
        f1 = f_gen.generate_function()

        def f(z):
            ans = f1(z)
            if np.isfinite(ans):
                return ans
            else:
                return 0

        d = d_c.Domain(-3, 3, -3, 3)
        f_p = d_c.FunctionPlot(f, d_c.domain_colouring, d, 512, 512, grid=False)
        f_p.save()
        done = True
    except Exception:
        pass
    if done is True:
        break

api.update_with_media('temp.png', status=f_gen.current_function_text)
