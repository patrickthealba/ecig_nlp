# ecig_nlp
Repository for the code and aggregate data associated with our work to identify documentation of electronic cigarette use and described in the paper "Development of a Natural Language Processing System to Identify Clinical Documentation of Electronic Cigarette Use"

A summary of the initial Models tested is found in the Directory "Model_Training", the batch processing script and relevant regex keyterms used is found in the directory "processing".  

## Data

The original data used for this model contains PHI and cannot be shared outside of the VA firewall.  However, a synthetic set of examples and resulting examples are found in the Data folder.  


Code
Here is how to run the code for baselines and deep learning components.

## Processing Complete Documents

![Alt Image text](https://github.com/patrickthealba/ecig_nlp/blob/master/processing/Figure%201.jpg?raw=true "Figure 1 Processing Pipeline")


Our cohort consists of more than 12 million veterans with over 4 billion clinical documents. It is challenging to process all these documents with limited computing resources. To address this challenge we implement a workflow shown in figure 1 which tackles this problem in four main steps: 

1) All clinical documents are stored in Microsoft SQL Server and are full text indexed. We first use full text index search with a manually developed list of key terms which largely covers smoking and ENDS. This reduces cohort documents to about 45 million. 

2) We then use a manually developed list of regular expressions of ENDS key terms to process all these 45 million documents. This resulted about 3.5 million documents that contains ENDS key terms. – see Supplementary Materials for complete term sets* 

3) The system next extracts ENDS key terms and surrounding context for each of these 3.5 million documents. 

4) The system finally classifies the context into ENDS usages with deep learning algorithms. All instances of an extracted term were classified into one of five categories: Active-User, Usage-Unknown, Irrelevant, Former-User, and Non-User. 

In the first and second step, all key terms associated with ENDS terminology act as the basis for the ENDS name entity recognition (NER) identified by the NLP system. The process of identifying relevant key terms can be found in previously published work and are also available in this project's git. After ENDS related keywords are identified the surrounding context is extracted for classification, including the previous sentence, the sentence in which the ENDS key word is identified, and the following sentence, if available. 

The end-to-end NLP pipeline (Figure 1) with the final ClinicalBert classification model was implemented with PySpark on a local machine with a CPU of 128 logical cores and 2 TeraByte Memory which took approximately 4 days for extracting and inference.  
