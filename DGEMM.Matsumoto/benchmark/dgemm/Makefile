include ../Makefile.in

TARGET	= bench_dgemm

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) $< ../auxiliary/libauxiliary.a $(LDFLAGS) -o $@

clean:
	$(RM) $(TARGET)

test: all
	$(RUNOPTION) ./$(TARGET) 0 0 0  1536  71 1
	$(RUNOPTION) ./$(TARGET) 0 0 1  1536  71 1
	$(RUNOPTION) ./$(TARGET) 0 1 0  1536  71 1
	$(RUNOPTION) ./$(TARGET) 0 1 1  1536  71 1
