##
# Script for running sensitivity analyses over R, K, and s. Comment out
# chunks below to run analysis of a subset of all three.
#

## EFFICIENCY OF COVERT SIGNALING ##

for R in 0.33 0.66; do
    printf "\nSubmitting basic receptivity experiment part 1\n"
    fname=data/"basic-$R"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R$R  \
        -qstd.q -j$fname -n24 -t"06:00:00"

    printf "\nSubmitting basic receptivity experiment part 2\n"
    fname=data/"basic-$R"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R$R  \
        -qstd.q -j$fname -n24 -t"06:00:00"

    ## DISLIKING ##

    printf "\nSubmitting basic disliking experiment part 1\n"
    fname=data/"basic-$R"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R$R  \
        -qstd.q -j$fname -n24 -t"06:00:00"

    printf "\nSubmitting basic disliking experiment part 2\n"
    fname=data/"basic-$R"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R$R  \
        -qstd.q -j$fname -n24 -t"06:00:00"
done


# Ugly to just copy and paste, but it's sensitivity analysis time, 
# time to git r dun.
for K in 3 15 21; do
    printf "\nSubmitting basic receptivity experiment part 1\n"
    fname=data/"basic-K=$K"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -K$K  \
        -qstd.q -j$fname -n24 -t"06:00:00"

    printf "\nSubmitting basic receptivity experiment part 2\n"
    fname=data/"basic-K=$K"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -K$K  \
        -qstd.q -j$fname -n24 -t"06:00:00"

    ## DISLIKING ##

    printf "\nSubmitting basic disliking experiment part 1\n"
    fname=data/"basic-K=$K"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -K$K  \
        -qstd.q -j$fname -n24 -t"06:00:00"

    printf "\nSubmitting basic disliking experiment part 2\n"
    fname=data/"basic-K=$K"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -K$K  \
        -qstd.q -j$fname -n24 -t"06:00:00"
done


# Ugly to just copy and paste, but it's sensitivity analysis time, 
# time to git r dun.
for similarity_benefit in 0.1 0.5 1.0; do
    printf "\nSubmitting basic receptivity experiment part 1\n"
    fname=data/"basic-s=$similarity_benefit"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv  \
        -qstd.q -j$fname -n24 -t"06:00:00" --similarity_benefit=$similarity_benefit

    printf "\nSubmitting basic receptivity experiment part 2\n"
    fname=data/"basic-s=$similarity_benefit"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv  \
        -qstd.q -j$fname -n24 -t"06:00:00" --similarity_benefit=$similarity_benefit

    ## DISLIKING ##

    printf "\nSubmitting basic disliking experiment part 1\n"
    fname=data/"basic-s=$similarity_benefit"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv  \
        -qstd.q -j$fname -n24 -t"06:00:00" --similarity_benefit=$similarity_benefit

    printf "\nSubmitting basic disliking experiment part 2\n"
    fname=data/"basic-s=$similarity_benefit"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv  \
        -qstd.q -j$fname -n24 -t"06:00:00" --similarity_benefit=$similarity_benefit
done
