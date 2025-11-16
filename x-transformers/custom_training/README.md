Phil Wang is using model_size:data = 1:5

19M Model -> req 19M x 4 = 380M tokens data
Under batch size 4, seq len 1000, number of batches = 19M x 4 / (4 x 1000) roughly 1e5 steps

64M Model -> req 64M x 4 = 1280M tokens data
Under batch size 4, seq len 1000, number of batches = 1280M x 4 / (4 x 1000) roughly 1.3 x 1e6 steps

Usually GPU is the bottleneck, not vram. 

constants
19M -> req 380M tokens data
post_fix += "_19M"
MODEL_DIM = 512
MODEL_DEPTH = 6
MODEL_HEADS = 8

57M -> req 1140M tokens data
post_fix += "_57M"
MODEL_DIM = 768
MODEL_DEPTH = 8
MODEL_HEADS = 12

64M -> req 1280M tokens data
post_fix += "_64M"
MODEL_DIM = 640
MODEL_DEPTH = 12
MODEL_HEADS = 10

151.3M -> req 3B tokens data, enwik9 (~1B) is not enough
post_fix += "_151.3M"
MODEL_DIM = 1024
MODEL_DEPTH = 12
MODEL_HEADS = 16