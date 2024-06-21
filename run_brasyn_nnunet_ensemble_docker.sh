docker run -it \
      --gpus=all \
      -v $PWD/pseudo_val_set:/workspace/inputs/ \
      -v $PWD/outputs:/workspace/outputs/  \
      --name test_nnunet \
      --ipc=host \
      --rm \
      winstonhutiger/brasyn_nnunet:ensemble  \
      /bin/bash -c "bash predict.sh"