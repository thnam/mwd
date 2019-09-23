CXX = g++
CXXFLAGS = -std=c++11 -g

CC = gcc
CCFLAGS = -std=c11 -g -Wall
LDFLAGS = -lm

NVCC = nvcc
CUFLAGS = -O3
CULIBS = -L/usr/local/cuda/lib64
CUINCS = -I/usr/local/cuda/include

CpuTarget = mwd
GpuTarget = gmwd
TestTarget = test

SRCDIR = ./srcs
OBJDIR = ./objs
INCS = -I$(SRCDIR)/
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))

CpuObj = $(OBJDIR)/$(CpuTarget).o
GpuSrcs = $(wildcard $(SRCDIR)/*.cu)
GpuObjs = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(GpuSrcs))

all: $(CpuTarget) $(GpuTarget) $(TestTarget)

$(CpuTarget): $(OBJS) $(CpuObj)
	$(CC) -o $@ $^ $(LDFLAGS) 

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(INCS) $(CCFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.c
	@mkdir -p $(OBJDIR)
	$(CC) $(INCS) $(CCFLAGS) -c $< -o $@

$(GpuTarget): $(GpuTarget).cu $(GpuObjs)
	$(NVCC) $(INCS) $(CUINCS) $(CUFLAGS) $(OBJS) $(GpuObjs) $< -o $@

$(TestTarget): $(TestTarget).cu $(GpuObjs)
	$(NVCC) $(INCS) $(CUINCS) $(CUFLAGS) $(OBJS) $(GpuObjs) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(INCS) -c $< $(CUFLAGS) -o $@

clean:
	rm -rf $(CpuTarget) $(OBJS) $(GpuTarget) $(TestTarget) $(GpuObjs)
