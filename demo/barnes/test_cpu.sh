rm log_cpu.txt
for i in 2 4 8 16 32 64
do
	./cpu.out -q 10240 -l $i >> log_cpu.txt 
done

grep -i time log_cpu.txt
