cat gslice_all.txt | sed -n '/Measurement_interval/p' | tail -n 50 | datamash -t , mean 4 pstdev 4
