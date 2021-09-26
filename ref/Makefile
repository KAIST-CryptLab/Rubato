XKCPPATH = XKCP_AVX2
XKCPHEADERPATH = $(XKCPPATH)/libXKCP.a.headers
XKCPHEADER = $(XKCPHEADERPATH)/KeccakP-1600-times4-SnP.h $(XKCPHEADERPATH)/SIMD256-config.h

CXXOPT = -O2 -I$(XKCPHEADERPATH) -I$(SEALHEADER) -std=c++17
LIBXKCP = -L$(XKCPPATH) -lXKCP

RUBATOSOURCE = Rubato.cpp ShakeAVX2.cpp
RUBATOHEADER = Rubato.h ShakeAVX2.h

all: rubato

debug: test_rubato.cpp $(RUBATOSOURCE) $(RUBATOHEADER) parms.h $(XKCPHEADER)
	$(CXX) -DDEBUG $(CXXOPT) test_rubato.cpp $(RUBATOSOURCE) $(LIBXKCP) -o test_rubato -mavx2

rubato: test_rubato.cpp $(RUBATOSOURCE) $(RUBATOHEADER) parms.h $(XKCPHEADER)
	$(CXX) -DNDEBUG $(CXXOPT) test_rubato.cpp $(RUBATOSOURCE) $(LIBXKCP) -o test_rubato -mavx2

clean:
	rm -f test_rubato
