python data_preprocess/manifold.py \
    --dataroot '' \
    --manifold '' \
    --simplify '' 

# dataroot:The root to single tooth before treatment
# manifold: Output path for manifold

python data_preprocess/simplify.py \
    --dataroot '' \
    --manifold '' \
    --simplify '' 

# simplify: Output path for simplify

python data_preprocess/datagen_maps.py \
    --dataroot '' \
    --output '' \

# dataroot：Ther root for single tooth after simplification
# output: Output root for data after remesh

