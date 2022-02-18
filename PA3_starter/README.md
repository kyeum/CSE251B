# CSE251B PA3
## Fully Convolutional Network for Semantic Segmentation

## PA3 Organization and Instruction
We used jupyter notebooks for the main body for each part.  


### Main Notebooks
1. Q3 was implemented in `PA3_3.ipynb`  
2. Q4a was implemented in `PA3_4a.ipynb`  
3. Q4b was implemented in `PA3_4b.ipynb`  
4. Q4b2 was implemented in `PA3_4b_2.ipynb`  
5. Q4b3 was implemented in `PA3_4b_3.ipynb`  
6. Q5a was implemented in `PA3_5a.ipynb`  
7. Q5b was implemented in `PA3_5b.ipynb`  
8. Q5c was implemented in `PA3_5c.ipynb`  
9. Visualization of generated segmentation mask of each experiment implemented in `Visualization.ipynb`  
10. Visualization of the 3 main data augmentation transforms in `VisualizeTransformations.ipynb`  

To run, simply open the jupyter notebooks with jupyter notebook and press `Run All`.

### Other files
`starter.py` is the starter running code for Q3.  
`starter_4.py` is the starter running code for Q4.  
`starter_5.py` is the starter running code for Q5 except for Q5b which uses `starter_4.py`.  
`dataloader.py` contains the data loading for Q3.  
`dataloader_4.py` contains the data loading and data augmentation for Q4. This is also used for Q5.  
`dataloder_4_test.py` contains the special dataloader used for the data augmentation visualization in `VisualizeTransformation.ipynb`.  
`basic_fcn.py` contains the model architecture used for Q3 and Q4.  
`custom_fcn.py` contains the model architecture used for Q5a.  
`tl_fcn.py` contains the model architecture used for Q5b.  
`unet_fcn.py` contains the model architecture used for Q5c.  
`utils.py` contains utility functions.  

### Plots
The generated plots for the report will be in the directory `plots/`.


