# Myeloma
comparing baselines -> comparaison d'algo de classification en utilisant tous les gènes

dim red -> 2 méthodes de réduction de dimension(pca,nmf)+ classif 

feature_select -> 2 méthodes de sélection (lasso / boruta) + classif

Results_bootstrap_DE.ipynb : Approche finale -> Analyse différentielle en prenant des sous-ensembles d'individus ensuite intersection entre les gènes sélectionnés 

selected_genes_analysis : analyse des gènes sélectionnés par la méthode finale (results_bootstrap_de) + projection umap des données[genes selectionnés] en 2 dim -> meilleur résultat
