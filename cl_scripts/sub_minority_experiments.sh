# NOTE r = 0.25 and R = 0.5 by default.


# K=3
# for similarity_threshold in 0.3 0.6 0.9; do 
#     for minority_trait_frac in 0.10 0.20; do 

#         fname=data/minority/$minority_trait_frac/part-`uuidgen`
#         subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 500 100 $fname.csv -R1.0  \
#             -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
#             -K$K -S$similarity_threshold 
#     done
# done


# K=9
# for similarity_threshold in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do 
#     for minority_trait_frac in 0.10 0.20; do 

#         fname=data/minority/$minority_trait_frac/part-`uuidgen`
#         subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 500 100 $fname.csv -R1.0  \
#             -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
#             -K$K -S$similarity_threshold 
#     done
# done


K=3
# for similarity_threshold in 0.3 0.5 0.8; do 
for similarity_threshold in 0.5; do 
    for minority_trait_frac in 0.10; do 
    # for minority_trait_frac in 0.10 0.20; do 

        fname=data/"minority-debug"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority-debug"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority-debug"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority-debug"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold
    done
done


K=9
# for similarity_threshold in 0.3 0.5 0.8; do 
for similarity_threshold in 0.8; do 
    for minority_trait_frac in 0.20; do 
    # for minority_trait_frac in 0.10 0.20; do 

        fname=data/"minority-debug"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority-debug"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority-debug"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold

        fname=data/"minority-debug"/$minority_trait_frac/part-`uuidgen`
        subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 100 25 $fname.csv -R1.0  \
            -qstd.q -j$fname -n24 -t"05:00:00" -m$minority_trait_frac \
            -K$K -S$similarity_threshold
    done
done


# K=9
# for similarity_threshold in 0.1 0.2 0.4 0.6 0.7; do 
#     for minority_trait_frac in 0.10 0.20; do 

#         fname=data/minority/$minority_trait_frac/part-`uuidgen`
#         subexp disliking 0.0:1.01:0.10 0.0:0.51:0.05 500 20 $fname.csv -R1.0  \
#             -qfast.q -j$fname -n24 -t"04:00:00" -m$minority_trait_frac \
#             -K$K -S$similarity_threshold 
#     done
# done
