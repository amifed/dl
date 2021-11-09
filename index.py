from os import PRIO_PGRP
import models.densenet as densenet

print(densenet.densenet121().__class__.__name__)
