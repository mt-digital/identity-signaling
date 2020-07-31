# Testing three values of K (cultural diversity) and ten similarity
# thresholds. This requires thirty directories be created in total,
# which can be done with commands
# mkdir -p output_data/tolerance_diversity/0.{1..9}/"K="{10,15,20}/
# mkdir -p output_data/tolerance_diversity/1.0/"K="{10,15,20}/


for K in 3 5 9 15 21; do
    for similarity_threshold in 0.{1..9} 1.0; do

        fname=data/tolerance_diversity/$similarity_threshold/"K=$K"/part-`uuidgen`
        subexp disliking 0.05:0.46:0.2 0.0:0.51:0.1 500 100 $fname.csv -R0.5  \
            -qfast.q -j$fname -n24 -t"04:00:00" -S$similarity_threshold -K$K
    done
done
