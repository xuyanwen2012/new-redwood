all:
	make sycl

cpu:
	g++ -o cpu.out -O3 --std=c++17 Main.cpp -lpthread

sycl:
	clang++ -o sycl.out -fsycl -O3 --std=c++17 RedwoodSycl.cpp Main.cpp -lpthread -DREDWOOD_BACKEND=2

clean:
	rm *.out *.o *.txt