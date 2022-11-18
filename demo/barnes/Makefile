all:
	make sycl

none:
	g++ -o none.out -O3 --std=c++17 RedwoodNone.cpp Main.cpp -lpthread

cpu:
	g++ -o cpu.out -O3 --std=c++17 RedwoodCpu.cpp Main.cpp -lpthread

sycl:
	clang++ -o sycl.out -fsycl -O3 --std=c++17 RedwoodSycl.cpp Main.cpp -lpthread

clean:
	rm *.out *.o *.txt