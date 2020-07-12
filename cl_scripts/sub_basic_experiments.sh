
printf "\nSubmitting basic receptivity experiment part 1\n"
fname=output_data/basic/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.05 0.0:0.51:0.05 500 10 $fname.csv -R0.5  \
    -qstd.q -j$fname -n24 -t"04:00:00"

printf "\nSubmitting basic receptivity experiment part 2\n"
fname=output_data/basic/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.05 0.0:0.51:0.05 500 10 $fname.csv -R0.5  \
    -qstd.q -j$fname -n24 -t"04:00:00"

printf "\nSubmitting basic disliking experiment part 1\n"
fname=output_data/basic/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 10 $fname.csv -R0.5  \
    -qstd.q -j$fname -n24 -t"04:00:00"

printf "\nSubmitting basic disliking experiment part 2\n"
fname=output_data/basic/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 10 $fname.csv -R0.5  \
    -qstd.q -j$fname -n24 -t"04:00:00"


