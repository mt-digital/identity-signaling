

## DISLIKING ##

# First part: set delta to be three different values and run basic
# experiment.
for delta in 0.05 0.45 0.75; do

    # Submit 100 trials for each setting by splitting into four sets of 
    # 25 trials.
    for _ in {1..4}; do
        printf "\nSubmitting basic disliking experiment part 1\n"
        fname=data/"basic"/delta/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            --two_disliking_penalty=$delta \
            -qstd.q -j$fname -n24 -t"10:00:00"
    done

done


# Next: vary delta over range with three different one-agent disliking
# penalty.
for delta in `seq 0.0 0.05 1.0`; do

    # Submit 100 trials for each setting by splitting into four sets of 
    # 25 trials.
    for _ in {1..4}; do
        printf "\nSubmitting basic disliking experiment part 1\n"
        fname=data/"basic"/delta/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.05:0.46:0.2 100 25 $fname.csv -R1.0  \
            --two_disliking_penalty=$delta \
            -qstd.q -j$fname -n24 -t"10:00:00"
    done

done
