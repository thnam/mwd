CXX = g++
CXXFLAGS = -std=c++11 -g
CC = gcc
CCFLAGS = -std=c11 -g -Wall
LDFLAGS = -lm
NVCC = nvcc
CUFLAGS = -g -G
CULIBS = -L/usr/local/cuda/lib64
CUINCS = -I/usr/local/cuda/include

CpuTarget = mwd
GpuTarget = gmwd

SRCDIR = ./srcs
OBJDIR = ./objs
INCS = -I$(SRCDIR)/
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))

CpuObj = $(OBJDIR)/$(CpuTarget).o

all: $(CpuTarget) $(GpuTarget)

$(CpuTarget): $(OBJS) $(CpuObj)
	$(CC) -o $@ $^ $(LDFLAGS) 

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(INCS) $(CCFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.c
	@mkdir -p $(OBJDIR)
	$(CC) $(INCS) $(CCFLAGS) -c $< -o $@

$(GpuTarget): $(GpuTarget).cu
	$(NVCC) $(INCS) $(CUINCS) $(CUFLAGS) $(OBJS) $< -o $@


clean:
	rm -rf $(CpuTarget) $(OBJS) $(GpuTarget)
