python -m pdb pairwise_celltype.py --h5ad_path "/data_volume/memento/hbec/HBEC_type_I_filtered_counts_deep.h5ad" \
    --ct_col leiden \
    --donor_col donor \
    --condition_col stim \
    --batch_col time