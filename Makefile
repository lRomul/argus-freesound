NAME?=argus-freesound
COMMAND?=bash
OPTIONS?=

GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

.PHONY: all
all: stop build run

.PHONY: build
build:
	docker build -t $(NAME) .

.PHONY: stop
stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

.PHONY: run
run:
	docker run --rm -dit \
		$(OPTIONS) \
		$(GPUS_OPTION) \
		--net=host \
		--ipc=host \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		$(COMMAND)
	docker attach $(NAME)

.PHONY: attach
attach:
	docker attach $(NAME)

.PHONY: logs
logs:
	docker logs -f $(NAME)

.PHONY: exec
exec:
	docker exec -it $(OPTIONS) $(NAME) $(COMMAND)

.PHONY: run-demo
run-demo:
	xhost local:root
	docker run --rm -it \
		$(OPTIONS) \
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
