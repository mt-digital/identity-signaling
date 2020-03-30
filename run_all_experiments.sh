# run_all_experiments.sh
# 2020-03-29

# An executable bit of documentation of how to run all the experiments used
# for the computational modeling publication.

# runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 40 4 invasion-part-`uuidgen`.csv -R0.5  \
fname=invasion-part-${uuidgen}
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.95

fname=invasion-part-${uuidgen}
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.95

fname=invasion-part-${uuidgen}
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.95

fname=invasion-part-${uuidgen}
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.5

fname=invasion-part-${uuidgen}
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.95

fname=invasion-part-${uuidgen}
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.5

# Get header from last filename
head -n1 $fname > invasion-full.csv
tail invasion-part-*.csv >> invasion-full.csv
