all:
	cd ../..; \
	sudo ~/dockcross-linux-riscv64 bash -c '$$CC -O3 -o ./demo/fpga-harversine-nn/duet.out demo/fpga-harversine-nn/Main.cpp -DREDWOOD_BACKEND=3 -DNDEBUG -std=c++20 -lstdc++ -lm -lpthread -static'; \
	cd demo/fpga-harversine-nn;

clean:
	rm *.out
