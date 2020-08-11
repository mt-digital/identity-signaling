
K=3
for similarity_threshold in 0.3 0.5 0.8; do 
# for similarity_threshold in 0.5; do 
    # for minority_trait_frac in 0.10; do 
    for minority_trait_frac in 0.10 0.20; do 

        fname=data/"minority"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold
    done
done


K=9
for similarity_threshold in 0.3 0.5 0.8; do 
# for similarity_threshold in 0.8; do 
    # for minority_trait_frac in 0.20; do 
    for minority_trait_frac in 0.10 0.20; do 

        fname=data/"minority"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold
    done
done

