##
# This script co-varies the two disliking penalties, d and delta. d 
# is the penalty when only one individual dislikes the other, and
# delta is the penalty when both agents dislike one another.
#
# Author: Matthew A. Turner
# Date: 10-18-2020
#

# First part: set delta to be three different values and run basic
# experiment.
for delta in 0.05 0.25 0.65 0.85; do

    # Submit 50 trials for each setting by splitting into five sets of 
    # 10 trials.
    for ii in {1..5}; do
        printf "\nSubmitting delta static experiment for delta=$delta, part $ii\n"
        fname=data/delta_static/delta\=$delta/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -R1.0  \
            --two_dislike_penalty=$delta \
            -qfast-ib.q -j$fname -n24 -t"04:00:00" 
    done

done


# Next: vary delta over range with three different one-agent disliking
# penalty.
for delta in `seq 0.0 0.1 1.0`; do
# for delta in 0.2; do

    # Submit 50 trials for each setting by splitting into five sets of 
    # 10 trials.
    # for ii in {1..5}; do
    for ii in {1..5}; do
        printf "\nSubmitting delta varies experiment for delta=$delta, part $ii\n"
        fname=data/delta_varies/delta\=$delta/part-`uuidgen`
        subexp disliking 0.05:0.86:0.2 0.0:0.51:0.05 100 10 $fname.csv -R1.0  \
            --two_dislike_penalty=$delta \
            -qfast-ib.q -j$fname -n24 -t"04:00:00"
    done

done


### XXX FOR FILLING IN SLOW 0.25 ONES, WHY SLOW??? XXX ###
# for delta in 0.25; do

#     # Submit 50 trials for each setting by splitting into five sets of 
#     # 10 trials.
#     for ii in {1..5}; do
#         printf "\nSubmitting delta static experiment for delta=$delta, part $ii\n"
#         fname=data/delta_static/delta\=$delta/part-`uuidgen`
#         subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 10 $fname.csv -R1.0  \
#             --two_dislike_penalty=$delta \
#             -qstd.q -j$fname -n24 -t"06:00:00" 
#     done

# done
