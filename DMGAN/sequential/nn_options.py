"""
Discription: A Python Progress Meter
Author: Nianzu Ethan Zheng
Date: 2018-1-21
Copyright
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from base_options import BaseConfig


class NetConfig(BaseConfig):
    def __init__(self):
        BaseConfig.__init__(self)
        self.gradient_clipping_lower = -0.1
        self.gradient_clipping_upper = 0.1
        self.assets_path = '../supervised/assets'


if __name__=='__main__':
    config = NetConfig()
    print(config.learning_rate)