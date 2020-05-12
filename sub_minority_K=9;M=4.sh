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





for i in {1..10}; do
    fname=output_data/minority_K9M4/part-`uuidgen`
    subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 10 $fname.csv -R0.5  \
        --n_minmaj_traits=4 -K9 -m0.10 \
        -qfast.q -j$fname -n24 -t"04:00:00" 
        # -qstd.q -j$fname -n20 -t"04:00:00" --n_minmaj_traits=4 -K10
done
