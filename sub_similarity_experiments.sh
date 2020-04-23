# * Set up simulations where agents are characterized by 10 
#     binary traits (instead of 3). 
# * Run simulations where "similarity" is defined as having at least X traits 
#     in common, where X = {1, 2, 3, 4, ..., 10}. We should find here 
#     that when X increases, we get more covert signaling. 
# * Run new majority-minority dynamics using a threshold of 50% similar. Set 
#     minority to be 10% of the population, where the first 4 traits are set to 
#     zero, and one for the majority (so that minority individuals really need 
#     to find each other). This will be a good confirmation of the trend we see 
#     with 3 traits. 
# * You can run all these with just one value of r/R and maybe 3 values of d.


# Experiment 1: 

for similarity_threshold in 0.{1..9} 1.0; do

    fname=output_data/minority-similarity/$similarity_threshold/part-`uuidgen`
    echo subexp disliking 0.05:0.46:0.2 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
        -qfast.q -j$fname -n24 -t"04:00:00" -S$similarity_threshold -K10

done


echo subexp disliking 0.05:0.46:0.2 0.0:0.51:0.1 500 50 $fname.csv -R0.5  \
    -qfast.q -j$fname -n24 -t"04:00:00" --n_minmaj_traits=4 -K10
