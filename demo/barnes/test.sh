rm log.txt
for i in 2 4 8 16 32 64 128 256 512
do
	./sycl.out -q 10240 -l $i >> log.txt 
done
grep -i time log.txt

