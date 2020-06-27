##
# Script to run missing parameter settings for previous minority experiment.
# Date: 6/27/20
#

# Submit missing trials for K=3, rho_minor=0.1.
for similarity_threshold in 0.3 0.4 0.5; do
    K=3
    minority_trait_frac=0.1
    fname=output_data/minority_supp/$minority_trait_frac/part-`uuidgen`
    subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 50 $fname.csv -R0.5  \
        -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
        -K$K -S$similarity_threshold 
done

# Submit missing trials for K=3, rho_minor=0.2.
for similarity_threshold in 0.3 0.5 0.7; do
    K=3
    minority_trait_frac=0.2
    fname=output_data/minority_supp/$minority_trait_frac/part-`uuidgen`
    subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 50 $fname.csv -R0.5  \
        -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
        -K$K -S$similarity_threshold 
done

# Only one failed for K=9 across both similarity thresholds; no need to loop.
K=9
similarity_threshold=0.4
minority_trait_frac=0.2

fname=output_data/minority_supp/$minority_trait_frac/part-`uuidgen`
subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 50 $fname.csv -R0.5  \
    -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
    -K$K -S$similarity_threshold 
