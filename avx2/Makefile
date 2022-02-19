XKCPPATH = XKCP_AVX2
XKCPHEADERPATH = $(XKCPPATH)/libXKCP.a.headers
XKCPHEADER = $(XKCPHEADERPATH)/KeccakHash.h

COPT = -O3 -march=native
CXXOPT = -O3 -march=native -I$(XKCPHEADERPATH) -std=c++17
LIBXKCP = -L$(XKCPPATH) -lXKCP
LIBOPENSSL = -lcrypto -lssl

RUBATOSOURCE = Rubato.cpp xof_aes.cpp xof_shake.cpp util.c
RUBATOHEADER = Rubato.h xof_aes.h xof_shake.h parms.h util.h

all: main

bench: bench_rubato.cpp $(RUBATOSOURCE) $(RUBATOHEADER) $(XKCPHEADER)
	gcc $(COPT) util.c -c -o util.o
	$(CXX) $(CXXOPT) bench_rubato.cpp -c -o bench.o
	$(CXX) $(CXXOPT) Rubato.cpp -c -o rubato.o
	$(CXX) $(CXXOPT) xof_aes.cpp $(LIBOPENSSL) -c -o xof_aes.o
	$(CXX) $(CXXOPT) xof_shake.cpp $(LIBXKCP) -c -o xof_shake.o
	$(CXX) $(CXXOPT) bench.o rubato.o xof_aes.o xof_shake.o util.o $(LIBXKCP) $(LIBOPENSSL) -o bench_rubato
	rm -f bench.o rubato.o xof_aes.o xof_shake.o util.o

test: test_rubato.cpp $(RUBATOSOURCE) $(RUBATOHEADER) $(XKCPHEADER)
	gcc $(COPT) util.c -c -o util.o
	$(CXX) $(CXXOPT) Rubato.cpp -c -o rubato.o
	$(CXX) $(CXXOPT) xof_aes.cpp $(LIBOPENSSL) -c -o xof_aes.o
	$(CXX) $(CXXOPT) xof_shake.cpp $(LIBXKCP) -c -o xof_shake.o
	$(CXX) $(CXXOPT) test_rubato.cpp $(LIBXKCP) -c -o test_rubato.o
	$(CXX) $(CXXOPT) test_rubato.o rubato.o xof_aes.o xof_shake.o util.o $(LIBXKCP) $(LIBOPENSSL) -o test_rubato
	rm -f test_rubato.o rubato.o xof_aes.o xof_shake.o util.o

main: main.cpp $(RUBATOSOURCE) $(RUBATOHEADER) $(XKCPHEADER)
	gcc $(COPT) util.c -c -o util.o
	$(CXX) $(CXXOPT) main.cpp -c -o main.o
	$(CXX) $(CXXOPT) Rubato.cpp -c -o rubato.o
	$(CXX) $(CXXOPT) xof_aes.cpp $(LIBOPENSSL) -c -o xof_aes.o
	$(CXX) $(CXXOPT) xof_shake.cpp $(LIBXKCP) -c -o xof_shake.o
	$(CXX) $(CXXOPT) main.o rubato.o util.o xof_aes.o xof_shake.o $(LIBXKCP) $(LIBOPENSSL) -o main
	rm -f main.o rubato.o xof_aes.o xof_shake.o util.o

clean:
	rm -f *.o bench_rubato test_rubato main
