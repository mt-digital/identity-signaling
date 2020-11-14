
# figure_dir='scratch_figures'
figure_dir='../Papers/covert-signaling-overleaf/Figures/components'

for analysis in basic invasion minority similarity_threshold \
    r_k_s_sensitivity delta_sensitivity; do

    run_analysis $analysis --figure_dir=$figure_dir/$analysis

done
