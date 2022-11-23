all:
	make sycl
	
cpu:
	g++ -o cpu.out -O3 --std=c++17 RedwoodCpu.cpp MainCpu.cpp -lpthread

sycl:
	clang++ -o sycl.out -fsycl -O3 --std=c++17 RedwoodSycl.cpp Main.cpp -lpthread

clean:
	rm *.out *.o *.txt