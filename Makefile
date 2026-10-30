CFLAGS ?= -O2 -g -Wall -Wextra

all: libpager.so libnopopulate.so test/test_reader

libpager.so: libpager.c
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $< -ldl -lpthread

libnopopulate.so: libnopopulate.c
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $< -ldl

test/test_reader: test/test_reader.c
	$(CC) $(CFLAGS) -o $@ $< -lpthread

test: all
	cd test && python3 gen_fake_gguf.py fake.gguf
	set -e; cd test; for cfg in sync:256 parallel:256 async:256 async:4; do \
	  io=$${cfg%:*}; blk=$${cfg#*:}; echo "== PAGER_IO=$$io PAGER_BLOCK_KB=$$blk"; \
	  export PAGER_IO=$$io PAGER_BLOCK_KB=$$blk PAGER_MIN_MB=0 LD_PRELOAD=../libpager.so; \
	  PAGER_MODE=trace PAGER_LOG=trace.csv ./test_reader fake.gguf; \
	  PAGER_MODE=fifo PAGER_BUDGET_MB=1 PAGER_LOG=fifo.csv ./test_reader fake.gguf; \
	  PAGER_MODE=fifo PAGER_BUDGET_MB=1 PAGER_DIRECT=0 PAGER_LOG=fifo_nodirect.csv ./test_reader fake.gguf; \
	  unset LD_PRELOAD; \
	done
	python3 simulate.py test/trace.csv --budgets-mb 0.5,1,2 --skip 0

clean:
	rm -f libpager.so libnopopulate.so test/test_reader test/fake.gguf* test/*.csv

.PHONY: all test clean
