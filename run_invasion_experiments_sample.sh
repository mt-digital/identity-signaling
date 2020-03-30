# run_all_experiments.sh
# 2020-03-29

# An executable bit of documentation of how to run all the experiments used
# for the computational modeling publication.

printf "\nRunning .95, .95\n"
fname=invasion-part-`uuidgen`
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.95

printf "\nRunning .5, .95\n"
fname=invasion-part-`uuidgen`
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.95

printf "\nRunning .05, .95\n"
fname=invasion-part-`uuidgen`
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.95

printf "\nRunning .95, .5\n"
fname=invasion-part-`uuidgen`
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.5

printf "\nRunning .5, .5\n"
fname="invasion-part-`uuidgen`"
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.5

printf "\nRunning .05, .95\n"
fname=invasion-part-`uuidgen`
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.5

printf "\nRunning .95, .05\n"
fname=invasion-part-`uuidgen`
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.05

printf "\nRunning .5, .05\n"
fname=invasion-part-`uuidgen`
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.05

printf "\nRunning .05, .05\n"
fname=invasion-part-`uuidgen`
runexp disliking 0.0:0.50:0.25 0.0:0.50:0.25 40 4 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.05

# Get header from last filename
head -n1 $fname.csv > invasion-full.csv
tail -n+1 invasion-part-*.csv >> invasion-full.csv
