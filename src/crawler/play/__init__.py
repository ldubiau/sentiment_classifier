# -*- coding: utf-8 -*-

import logging
import sys
import os

__author__ = 'Luciana Dubiau'

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
output_path = os.path.join(base_path, 'data2')
app_ids = [
#"com.hoyts",
#"com.mobilenik.mobilebanking.individuos", 
#"com.dropbox.android",
#"ar.com.mobatio.lanacion.club",
#"org.microemu.android.model.common.VTUserApplicationLINKMB",
#"com.mobisystems.editor.office_registered",
#"ar.com.guiaoleo.activity"
#"com.google.android.apps.translate"
#"com.gm.despegar",
#"ar.gob.buenosaires.comollego"
#"com.overflow.cinemark.activity"
#"com.google.android.apps.finance",
#"com.google.android.gm",
#"com.google.android.music"
#"com.google.android.googlequicksearchbox"
#"com.mercadolibre" -> page 244
#"com.king.candycrushsaga" -> page 222
]

logger = logging.getLogger(__package__)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
