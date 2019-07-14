NAME=argus-freesound

.PHONY: all build stop run 

all: stop build run

build:
	docker build -t $(NAME) .

stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

run:
	nvidia-docker run --rm -it \
		--net=host \
		--ipc=host \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		bash

run-demo:
	xhost local:root
	nvidia-docker run --rm -it \
		--net=host \
		--ipc=host \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(HOME)/.Xauthority:/root/.Xauthority \
		-e DISPLAY=$(shell echo ${DISPLAY}) \
		--device /dev/snd:/dev/snd \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		python demo.py
