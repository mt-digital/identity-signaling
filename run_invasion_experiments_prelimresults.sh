# run_all_experiments.sh
# 2020-03-29

# An executable bit of documentation of how to run all the experiments used
# for the computational modeling publication.

printf "\nRunning .95, .95\n"
fname=data/scratch/invasion-part-`uuidgen`
runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 100 10 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.95 > $fname.log

printf "\nRunning .5, .95\n"
fname=data/scratch/invasion-part-`uuidgen`
runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 100 10 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.95 > $fname.log

printf "\nRunning .05, .95\n"
fname=data/scratch/invasion-part-`uuidgen`
runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 100 10 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.95 > $fname.log

printf "\nRunning .95, .5\n"
fname=data/scratch/invasion-part-`uuidgen`
runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 100 10 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.5 > $fname.log

printf "\nRunning .5, .5\n"
fname="data/scratch/invasion-part-`uuidgen`"
runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 100 10 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.5 > $fname.log

printf "\nRunning .05, .95\n"
fname=data/scratch/invasion-part-`uuidgen`
runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 100 10 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.5 > $fname.log

printf "\nRunning .95, .05\n"
fname=data/scratch/invasion-part-`uuidgen`
runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 100 10 $fname.csv -R0.5  \
    --initial_prop_covert=0.95 --initial_prop_churlish=0.05 > $fname.log

printf "\nRunning .5, .05\n"
fname=data/scratch/invasion-part-`uuidgen`
runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 100 10 $fname.csv -R0.5  \
    --initial_prop_covert=0.5 --initial_prop_churlish=0.05 > $fname.log

printf "\nRunning .05, .05\n"
fname=data/scratch/invasion-part-`uuidgen`
runexp disliking 0.0:0.51:0.25 0.0:0.51:0.25 100 10 $fname.csv -R0.5  \
    --initial_prop_covert=0.05 --initial_prop_churlish=0.05 > $fname.log

# Get header from last filename
head -n1 $fname.csv > data/scratch/invasion-full.csv
tail -q -n+2 data/scratch/invasion-part-*.csv >> data/scratch/invasion-full.csv
