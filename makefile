# Name of the executable
TARGET ?= main_linsolve

# Directory of the target's source file
TARGET_DIR ?= demo

# Additional compiler flags (e.g. EXTRA_CFLAGS="-O3 -fopt-info-vec=vec.log")
EXTRA_CFLAGS ?=

CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic $(EXTRA_CFLAGS)
LFLAGS = -lm

SRC = src/detectum.c $(TARGET_DIR)/$(TARGET).c
INC = -I inc

.PHONY: all

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INC) $(SRC) $(LFLAGS) -o $@
