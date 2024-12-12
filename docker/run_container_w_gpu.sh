sudo docker run -it \
-v "$(pwd)/../":/workspace \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-e XDG_RUNTIME_DIR=/tmp/runtime-root \
-e DISPLAY=unix$DISPLAY \
--net=host \
--gpus all \
--ipc=host \
--privileged \
--name hf2_vad_ad \
leolixingyou/hf2_vad_ad_test:ros1_v0.2
