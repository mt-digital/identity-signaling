## EFFICIENCY OF COVERT SIGNALING ##

# printf "\nSubmitting basic receptivity experiment part 1\n"
# fname=data/"basic"/receptivity/part-`uuidgen`
# subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
#     -qstd.q -j$fname -n24 -t"06:00:00"

# printf "\nSubmitting basic receptivity experiment part 2\n"
# fname=data/"basic"/receptivity/part-`uuidgen`
# subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
#     -qstd.q -j$fname -n24 -t"06:00:00"

# printf "\nSubmitting basic receptivity experiment part 3\n"
# fname=data/"basic"/receptivity/part-`uuidgen`
# subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
#     -qstd.q -j$fname -n24 -t"06:00:00"

# printf "\nSubmitting basic receptivity experiment part 4\n"
# fname=data/"basic"/receptivity/part-`uuidgen`
# subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
#     -qstd.q -j$fname -n24 -t"06:00:00"


## DISLIKING ##

printf "\nSubmitting basic disliking experiment part 1\n"
fname=data/"basic"/disliking/part-`uuidgen`
subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
    -qstd.q -j$fname -n24 -t"06:00:00"

printf "\nSubmitting basic disliking experiment part 2\n"
fname=data/"basic"/disliking/part-`uuidgen`
subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
    -qstd.q -j$fname -n24 -t"06:00:00"


printf "\nSubmitting basic disliking experiment part 3\n"
fname=data/"basic"/disliking/part-`uuidgen`
subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
    -qstd.q -j$fname -n24 -t"06:00:00"

printf "\nSubmitting basic disliking experiment part 4\n"
fname=data/"basic"/disliking/part-`uuidgen`
subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
    -qstd.q -j$fname -n24 -t"06:00:00"
