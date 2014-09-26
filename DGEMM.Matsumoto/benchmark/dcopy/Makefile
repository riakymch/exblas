include ../Makefile.in

TARGET	= bench_dcopy

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) $< ../auxiliary/libauxiliary.a $(LDFLAGS) -o $@

clean:
	$(RM) $(TARGET) $(TARGET).o

test: all
	./$(TARGET) 1 1 512 32 1
	./$(TARGET) 1 3 512 32 1
	./$(TARGET) 3 1 512 32 1
	./$(TARGET) 3 3 512 32 1
