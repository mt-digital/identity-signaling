# NOTE r = 0.25 and R = 0.5 by default.

for K in 3 9; do #5 9 15; do # 21; do
    for similarity_threshold in 0.{1..9} 1.0; do 
        # for minority_trait_frac in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45; do
        for minority_trait_frac in 0.10 0.20; do  # 0.25 0.30 0.35 0.40 0.45; do

            fname=output_data/minority_supp/$minority_trait_frac/part-`uuidgen`
            subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 50 $fname.csv -R0.5  \
                -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
                -K$K -S$similarity_threshold 

            fname=output_data/minority_supp/$minority_trait_frac/part-`uuidgen`
            subexp disliking 0.0:0.51:0.05 0.0:0.51:0.05 500 50 $fname.csv -R0.5  \
                -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
                -K$K -S$similarity_threshold 

        done
    done
done
