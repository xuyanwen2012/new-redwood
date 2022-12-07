rm log_cpu.txt
for i in 1 2 4 8 16 32 64 128
do
	./cpu.out -q 204800 -l $i >> log_cpu.txt 
done

grep -i time log_cpu.txt
