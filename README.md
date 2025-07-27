# From Outliers to Topics in Language Models: Anticipating Trends in News Corpora

This repository contains the results, data, visualizations, and code accompanying our paper submitted to [ICNLSP 2025](https://www.icnlsp.org/2025welcome/). 

## Repository Structure

### `DATA`


Contains all input data used for the analysis of both the TP French dataset and the GHG English dataset:
- **Dataset Embeddings**: 
  - Precomputed embeddings from 9 selected language models for both TP and GHG in `.pkl` files (`TP_EMBEDDINGS` and `GHG_EMBEDDINGS` folders).
  - Timeframe-specific embeddings in `.csv` files (`TP_EMBEDDINGS_TIMEFRAME` and `GHG_EMBEDDINGS_TIMEFRAME` folders).
  - Data in raw format can be provided upon request.

<!--
#### Extracts:
##### From TP dataset :
> *"L’École polytechnique a donné son accord pour l’installation d’un nouveau centre d’innovation et de recherche sur les énergies décarbonées ..."* — **Polytechnique**  
> *"Ce projet de nouveau centre est emblématique pour Total. Il place le Groupe au cœur d’un écosystème mondial d’innovation..."* — **TotalEnergies**  
> *"...la neutralité scientifique de la formation est menacée. Un pas de plus, hautement révélateur, dans la privatisation rampante de l’enseignement supérieur..."* — **Mediapart**  
> *"Au-delà de la problématique fondamentale de la neutralité de l’enseignement public, dans un contexte où Total cherche à maintenir ses investissements dans les énergies fossiles..."* — **Greenpeace**

##### From GHG dataset :
> *"President Biden's administration has proposed regulations to speed the transition to electric vehicles, committed $1 billion to help poor countries fight climate change and prepared what could be the first limits on greenhouse gas emissions from power plants."* — **The New York Times**  
> *"Nor is there any apparent scientific research that supports a claim that there are currently 15 million people dying yearly due to greenhouse gas emissions or any other single cause of death that is tracked..."* — **Fox News**  
> *"Low-income countries, which generate only a tiny fraction of global emissions, will experience the vast majority of deaths and displacement from the worst-case warming scenarios, the IPCC warns."* — **The Washington Post**  
> *"A lot of the world's resources, particularly international finance, goes toward reducing greenhouse emissions, which is known as mitigation. At the COP26 climate talks in Glasgow, Scotland, last year..."* — **CNN**
-->


---

### `RESULTS`
Contains the output data from the analysis of both datasets:
- **Section 4.2 & 5.2**: Topic Modeling results, including Silhouette score analysis by timeframe and model.
- **Section 4.3 & 5.3**: Analysis of outlier-to-topic transitions over time, including inter-model agreement ratios and alpha($\mathcal{a}$) for the 9 models.
- **Section 4.4 & 5.4**: 
  - TF-IDF Lexicon Delta analysis between outliers verifying hypothesis $\mathcal{H}$ and those not verifying $\mathcal{H}$.
  - Spearman correlation analysis between VADER neutrality scores, TextBlob subjectivity scores, and Delta TF-IDF values.
  - Stylometry reports based on [EnzoFleur's repository](https://github.com/EnzoFleur/style_embedding_evaluation).
  - Statistical significance tests for stylistic features between outlier classes.
---

### `CODE`
Contains Python scripts for the full experimental setup, organized by section:
- **4.2 & 5.2 Topic-Based Clustering**:
  - Silhouette score analysis by corpus (Tables 2 and 5).
  - Timeframe scatter plots for TP (Figure 1) and GHG (Figure 4).
- **4.3 & 5.3 Outlier Behavior**:
  - Hypothesis validation bar plots per model (Figures 2 and 5).
  - Proportion of outliers converting to clusters or remaining as outliers per Model (Tables 3 and 6).
  - Longitudinal analysis of outlier behavior and trend shifts.
- **4.4 & 5.4 Explanation**:
  - TF-IDF score Delta between outliers classes.
  - Subjectivity and neutrality Spearman correlation analysis.
  - Stylometric analysis box plots comparing outlier classes.
  - Statistical significance results (Figures 3 and 6).

---

### `PLOTS`
Contains visual representations of key findings:
- Timeframe scatter plots for TP and GHG datasets.
- Hypothesis validation bar plots by model.
- Stylometric analysis plots comparing outlier classes, including statistical significance results.
