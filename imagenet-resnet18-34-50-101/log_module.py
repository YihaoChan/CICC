# -*- coding: UTF-8 -*-
import logging
from logging import handlers
import os
from parameter import get_parameter


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info'):
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()  # 屏幕
        th = handlers.TimedRotatingFileHandler(when='D', filename=filename, encoding='utf-8')  # 文件
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


args = get_parameter()

log_dir = args.log_path
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log = Logger(os.path.join(log_dir, './%s.log' % args.network), level='info')
