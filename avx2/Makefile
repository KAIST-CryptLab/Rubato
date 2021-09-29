XKCPPATH = XKCP_AVX2
XKCPHEADERPATH = $(XKCPPATH)/libXKCP.a.headers
XKCPHEADER = $(XKCPHEADERPATH)/KeccakP-1600-SnP.h

COPT = -O3 -march=native
CXXOPT = -O3 -march=native -I$(XKCPHEADERPATH) -std=c++17
LIBXKCP = -L$(XKCPPATH) -lXKCP
LIBOPENSSL = -lcrypto -lssl

CPUNUM=2

all: main

bench: bench_rubato.cpp Rubato.cpp util.o parms.h xof_aes.cpp xof_shake.cpp
	for XOF in 0 ; do\
		for PARAM in 0 1 2 3 4 5 ; do\
			  $(CXX) -DXOF_TYPE=$$XOF -DPARAM_ID=$$PARAM $(CXXOPT) bench_rubato.cpp -c -o bench.o ;\
				$(CXX) -DXOF_TYPE=$$XOF -DPARAM_ID=$$PARAM $(CXXOPT) Rubato.cpp -c -o rubato.o ;\
				$(CXX) $(CXXOPT) -DXOF_TYPE=0 xof_aes.cpp $(LIBOPENSSL) -c -o xof.o ;\
			  $(CXX) -DXOF_TYPE=$$XOF -DPARAM_ID=$$PARAM $(CXXOPT) bench.o util.o rubato.o xof.o $(LIBXKCP) $(LIBOPENSSL) -o bench_rubato; \
				/usr/local/bin/boost $(CPUNUM) > /dev/null ;\
				taskset --cpu-list $(CPUNUM) ./bench_rubato >> result ;\
				/usr/local/bin/cooldown $(CPUNUM) > /dev/null ;\
		done ;\
	done
	for XOF in 1 2 ; do\
		for PARAM in 0 1 2 3 4 5 ; do\
			  $(CXX) -DXOF_TYPE=$$XOF -DPARAM_ID=$$PARAM $(CXXOPT) bench_rubato.cpp -c -o bench.o ;\
				$(CXX) -DXOF_TYPE=$$XOF -DPARAM_ID=$$PARAM $(CXXOPT) Rubato.cpp -c -o rubato.o ;\
				$(CXX) $(CXXOPT) -DXOF_TYPE=$$XOF xof_shake.cpp $(LIBOPENSSL) -c -o xof.o ;\
			  $(CXX) -DXOF_TYPE=$$XOF -DPARAM_ID=$$PARAM $(CXXOPT) bench.o util.o rubato.o xof.o $(LIBXKCP) $(LIBOPENSSL) -o bench_rubato; \
				/usr/local/bin/boost $(CPUNUM) > /dev/null ;\
				taskset --cpu-list $(CPUNUM) ./bench_rubato >> result ;\
				/usr/local/bin/cooldown $(CPUNUM) > /dev/null ;\
		done ;\
	done

main: main.cpp util.o Rubato.cpp parms.h xof_aes.cpp xof_shake.cpp
	$(CXX) $(CXXOPT) main.cpp -c -o main.o
	$(CXX) $(CXXOPT) Rubato.cpp -c -o rubato.o
	$(CXX) $(CXXOPT) xof_aes.cpp $(LIBOPENSSL) -c -o xof_aes.o
	$(CXX) $(CXXOPT) xof_shake.cpp $(LIBXKCP) -c -o xof_shake.o
	$(CXX) $(CXXOPT) main.o util.o rubato.o xof_aes.o xof_shake.o $(LIBXKCP) $(LIBOPENSSL) -o main

test: main.cpp util.o Rubato.cpp parms.h xof_aes.cpp xof_shake.cpp
	$(CXX) $(CXXOPT) test_rubato.cpp -c -o test.o
	$(CXX) $(CXXOPT) Rubato.cpp -c -o rubato.o
	$(CXX) $(CXXOPT) xof_aes.cpp $(LIBOPENSSL) -c -o xof_aes.o
	$(CXX) $(CXXOPT) xof_shake.cpp $(LIBXKCP) -c -o xof_shake.o
	$(CXX) $(CXXOPT) test.o util.o rubato.o xof_aes.o xof_shake.o $(LIBXKCP) $(LIBOPENSSL) -o test

util.o: util.c util.h
	gcc $(COPT) util.c -c -o util.o

clean:
	rm -f *.o test main bench_rubato_* result
