# -*- coding: utf-8 -*-

import logging
import sys
import os

__author__ = 'Luciana Dubiau'

LAST_RESTO_ID = 20000

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
output_path = os.path.join(base_path, 'data')

logger = logging.getLogger(__package__)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
