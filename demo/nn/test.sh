rm log.txt
for i in 32 64 128 256 512 1024 2048 4086
do
	./sycl.out -q 204800 -l $i >> log.txt 
done
grep -i time log.txt

