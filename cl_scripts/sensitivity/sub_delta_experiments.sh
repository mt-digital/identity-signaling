

## DISLIKING ##

# First part: set delta to be three different values and run basic
# experiment.
for delta in 0.05 0.45 0.65 0.85; do

    # Submit 50 trials for each setting by splitting into five sets of 
    # 10 trials.
    for _ in {1..5}; do
        printf "\nSubmitting basic disliking experiment part 1\n"
        fname=data/delta_static/delta\=$delta/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -R1.0  \
            --two_dislike_penalty=$delta \
            -qfast.q -j$fname -n24 -t"04:00:00"
    done

done


# Next: vary delta over range with three different one-agent disliking
# penalty.
for delta in `seq 0.0 0.1 1.0`; do

    # Submit 50 trials for each setting by splitting into five sets of 
    # 10 trials.
    for _ in {1..5}; do
        printf "\nSubmitting basic disliking experiment part 1\n"
        fname=data/delta_varies/delta\=$delta/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.05:0.86:0.2 100 10 $fname.csv -R1.0  \
            --two_dislike_penalty=$delta \
            -qfast.q -j$fname -n24 -t"04:00:00"
    done

done
