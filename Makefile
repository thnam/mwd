CXX = g++
CXXFLAGS = -std=c++11 -g
LDFLAGS =
CC = gcc
CCFLAGS = -std=c11 -g

TARGET = mwd


SRCDIR = ./srcs
OBJDIR = ./objs
INCS = -I$(SRCDIR)/
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))

SRCS += $(wildcard $(SRCDIR)/*.cc)
OBJS += $(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%.o,$(SRCS))
OBJS += $(OBJDIR)/$(TARGET).o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(LDFLAGS) $^ 

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	@mkdir -p $(OBJDIR)
	$(CXX) $(INCS) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(INCS) $(CCFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cc
	@mkdir -p $(OBJDIR)
	$(CXX) $(INCS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGET) $(OBJS)
