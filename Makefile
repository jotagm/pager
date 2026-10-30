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
	set -e; cd test; for io in sync parallel; do \
	  echo "== PAGER_IO=$$io"; \
	  PAGER_IO=$$io PAGER_MIN_MB=0 PAGER_MODE=trace PAGER_LOG=trace.csv LD_PRELOAD=../libpager.so ./test_reader fake.gguf; \
	  PAGER_IO=$$io PAGER_MIN_MB=0 PAGER_MODE=fifo PAGER_BUDGET_MB=1 PAGER_LOG=fifo.csv LD_PRELOAD=../libpager.so ./test_reader fake.gguf; \
	  PAGER_IO=$$io PAGER_MIN_MB=0 PAGER_MODE=fifo PAGER_BUDGET_MB=1 PAGER_DIRECT=0 PAGER_LOG=fifo_nodirect.csv LD_PRELOAD=../libpager.so ./test_reader fake.gguf; \
	done
	python3 simulate.py test/trace.csv --budgets-mb 0.5,1,2 --skip 0

clean:
	rm -f libpager.so libnopopulate.so test/test_reader test/fake.gguf* test/*.csv

.PHONY: all test clean
