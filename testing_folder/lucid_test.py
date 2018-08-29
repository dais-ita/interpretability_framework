import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
import lucid.optvis.render as render


model = models.InceptionV1()
model.load_graphdef()

_ = render.render_vis(model, "mixed4a_pre_relu:476")