## Docker build image
```
docker build . -t radtts_img
```

## Docker start
```
nvidia-docker run \
    --rm --gpus all \
    -v ${PWD}:/src \
    -it \
    radtts_img /bin/bash
```



## Inference command
```
export CONFIG_PATH=/src/configs/config_ljs_dap.json
export RADTTS_PATH=/src/models/radtts++ljs-dap.pt
export HG_PATH=/src/models/hifigan_libritts100360_generator0p5.pt
export HG_CONFIG_PATH=/src/models/hifigan_22khz_config.json
export TEXT_PATH=/src/sentences.txt

python inference.py \
-c ${CONFIG_PATH} \
-r ${RADTTS_PATH} \
-v ${HG_PATH} \
-k ${HG_CONFIG_PATH} \
-t ${TEXT_PATH} \
-s ljs \
--speaker_attributes ljs \
--speaker_text ljs \
-o results/
```


# Test torch installation inside container
```
import torch
torch.cuda.is_available()
torch.tensor([1.0, 2.0])
torch.tensor([1.0, 2.0]).cuda()
```