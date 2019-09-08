CXX = g++
CXXFLAGS = -std=c++11 -g
LDFLAGS =
CC = gcc
CCFLAGS = -std=c11 -g -Wall

TARGET = mwd


SRCDIR = ./srcs
OBJDIR = ./objs
INCS = -I$(SRCDIR)/
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))
OBJS += $(OBJDIR)/$(TARGET).o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(LDFLAGS) $^ 

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(INCS) $(CCFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.c
	@mkdir -p $(OBJDIR)
	$(CC) $(INCS) $(CCFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGET) $(OBJS)
