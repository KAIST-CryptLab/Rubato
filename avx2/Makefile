XKCPPATH = XKCP_AVX2
XKCPHEADERPATH = $(XKCPPATH)/libXKCP.a.headers
XKCPHEADER = $(XKCPHEADERPATH)/KeccakP-1600-SnP.h

COPT = -O2 -march=native
CXXOPT = -O2 -march=native -I$(XKCPHEADERPATH) -std=c++17
LIBXKCP = -L$(XKCPPATH) -lXKCP
LIBOPENSSL = -lcrypto -lssl

RUBATOSOURCE = Rubato.cpp xof_aes.cpp xof_shake.cpp util.c
RUBATOHEADER = Rubato.h xof_aes.h xof_shake.h parms.h util.h

all: rubato

bench: bench_rubato.cpp $(RUBATOSOURCE) $(RUBATOHEADER) $(XKCPHEADER)
	gcc $(COPT) util.c -c -o util.o
	$(CXX) $(CXXOPT) bench_rubato.cpp -c -o bench.o
	$(CXX) $(CXXOPT) Rubato.cpp -c -o rubato.o
	$(CXX) $(CXXOPT) xof_aes.cpp $(LIBOPENSSL) -c -o xof_aes.o
	$(CXX) $(CXXOPT) xof_shake.cpp $(LIBXKCP) -c -o xof_shake.o
	$(CXX) $(CXXOPT) bench.o rubato.o xof_aes.o xof_shake.o util.o cbd.o -o bench_rubato
	rm -f bench.o rubato.o xof_aes.o xof_shake.o util.o cbd.o

rubato: test_rubato.cpp $(RUBATOSOURCE) $(RUBATOHEADER) $(XKCPHEADER)
	gcc $(COPT) util.c -c -o util.o
	$(CXX) $(CXXOPT) Rubato.cpp -c -o rubato.o
	$(CXX) $(CXXOPT) xof_aes.cpp $(LIBOPENSSL) -c -o xof_aes.o
	$(CXX) $(CXXOPT) xof_shake.cpp $(LIBXKCP) -c -o xof_shake.o
	$(CXX) $(CXXOPT) test_rubato.cpp $(LIBXKCP) -c -o test_rubato.o
	$(CXX) $(CXXOPT) test_rubato.o rubato.o xof_aes.o xof_shake.o util.o $(LIBXKCP) $(LIBOPENSSL) -o test_rubato
	rm -f test_rubato.o rubato.o xof_aes.o xof_shake.o util.o

clean:
	rm -f test_rubato
