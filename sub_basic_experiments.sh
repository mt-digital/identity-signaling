
printf "\nSubmitting basic disliking experiment part 1\n"
fname=output_data/"basic_alpha=1"/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 20 $fname.csv -R0.5  \
    -qfast.q -j$fname -n24 -t"04:00:00" --learning_alpha=1.0

printf "\nSubmitting basic disliking experiment part 2\n"
fname=output_data/"basic_alpha=1"/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 20 $fname.csv -R0.5  \
    -qfast.q -j$fname -n24 -t"04:00:00" --learning_alpha=1.0


printf "\nSubmitting basic receptivity experiment part 1\n"
fname=output_data/"basic_alpha=1"/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.05 0.0:0.51:0.05 500 20 $fname.csv -R0.5  \
    -qfast.q -j$fname -n24 -t"04:00:00" --learning_alpha=1.0

printf "\nSubmitting basic receptivity experiment part 2\n"
fname=output_data/"basic_alpha=1"/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.05 0.0:0.51:0.05 500 20 $fname.csv -R0.5  \
    -qfast.q -j$fname -n24 -t"04:00:00" --learning_alpha=1.0
