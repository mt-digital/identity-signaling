# An executable bit of documentation of how to run all the experiments used
# for the computational modeling publication.

printf "\nRunning .95, .95\n"
fname=output_data/invasion/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.95 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .5, .95\n"
fname=output_data/invasion/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.95 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .05, .95\n"
fname=output_data/invasion/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.95 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .95, .5\n"
fname=output_data/invasion/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.5 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .5, .5\n"
fname=output_data/invasion/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.5 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .05, .95\n"
fname=output_data/invasion/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.5 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .95, .05\n"
fname=output_data/invasion/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.05 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .5, .05\n"
fname=output_data/invasion/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.05 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .05, .05\n"
fname=output_data/invasion/disliking/part-`uuidgen`
subexp disliking 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.05 \
    -qfast.q -j$fname -n24 -t"02:00:00"


############################################################
## RECEPTIVITY EXPERIMENT
############################################################

printf "\nRunning .95, .95\n"
fname=output_data/invasion/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.95 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .5, .95\n"
fname=output_data/invasion/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.95 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .05, .95\n"
fname=output_data/invasion/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.95 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .95, .5\n"
fname=output_data/invasion/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.5 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .5, .5\n"
fname=output_data/invasion/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.5 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .05, .95\n"
fname=output_data/invasion/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.5 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .95, .05\n"
fname=output_data/invasion/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.05 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .5, .05\n"
fname=output_data/invasion/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.05 \
    -qfast.q -j$fname -n24 -t"02:00:00"


printf "\nRunning .05, .05\n"
fname=output_data/invasion/receptivity/part-`uuidgen`
subexp receptivity 0.0:0.51:0.1 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.05 \
    -qfast.q -j$fname -n24 -t"02:00:00"
