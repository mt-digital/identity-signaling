##
# Script for running sensitivity analyses over R, K, and s. Comment out
# chunks below to run analysis of a subset of all three.
#

## EFFICIENCY OF COVERT SIGNALING ##

for R in 0.33 0.66; do
    ## DISLIKING ##
    printf "\nSubmitting overt signaling efficiency disliking experiment part 1\n"
    fname=data/"basic-overt_efficiency=$R"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -R$R  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00"

    printf "\nSubmitting overt signaling efficiency disliking experiment part 2\n"
    fname=data/"basic-overt_efficiency=$R"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -R$R  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00"

    ## COVERT RECEPTIVITY ##
    printf "\nSubmitting overt signaling efficiency receptivity experiment part 1\n"
    fname=data/"basic-overt_efficiency=$R"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -R$R  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00"

    printf "\nSubmitting overt signaling efficiency receptivity experiment part 2\n"
    fname=data/"basic-overt_efficiency=$R"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -R$R  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00"
done


# Ugly to just copy and paste, but it's sensitivity analysis time, 
# time to git r dun.
for K in 3 21; do
    printf "\nSubmitting K sensitivity receptivity experiment part 1\n"
    fname=data/"basic-K=$K"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -K$K  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00"

    printf "\nSubmitting K sensitivity receptivity experiment part 2\n"
    fname=data/"basic-K=$K"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -K$K  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00"

    ## DISLIKING ##

    printf "\nSubmitting K sensitivity disliking experiment part 1\n"
    fname=data/"basic-K=$K"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -K$K  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00"

    printf "\nSubmitting K sensitivity disliking experiment part 2\n"
    fname=data/"basic-K=$K"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -K$K  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00"
done


# Ugly to just copy and paste, but it's sensitivity analysis time, 
# time to git r dun.
for similarity_benefit in 0.1 1.0; do
    printf "\nSubmitting similarity benefit sensitivity receptivity experiment part 1\n"
    fname=data/"basic-s=$similarity_benefit"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00" --similarity_benefit=$similarity_benefit

    printf "\nSubmitting similarity benefit sensitivity receptivity experiment part 2\n"
    fname=data/"basic-s=$similarity_benefit"/receptivity/part-`uuidgen`
    subexp receptivity 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00" --similarity_benefit=$similarity_benefit

    ## DISLIKING ##

    printf "\nSubmitting similarity benefit sensitivity disliking experiment part 1\n"
    fname=data/"basic-s=$similarity_benefit"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00" --similarity_benefit=$similarity_benefit

    printf "\nSubmitting similarity benefit sensitivity disliking experiment part 2\n"
    fname=data/"basic-s=$similarity_benefit"/disliking/part-`uuidgen`
    subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv  \
        -qfast-ib.q -j$fname -n24 -t"04:00:00" --similarity_benefit=$similarity_benefit
done
