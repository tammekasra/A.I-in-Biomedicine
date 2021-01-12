<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# A.I - Protein Folding Algorithm

Final project for the Building AI course

## Summary

Humans have approximately 25,000 individual genes. Each one of them encodes seperate proteins and that have a distinct functions. One of the functions are to maintain a stable cell growth, cell division, communcations between cells and maintaining cell integrity. These complex functions keep organisms alive and helps them to evolve.
However, whenever these systems start malfunctioning, malicious cells can be formed: cancer, myeloma in Alzheimers, abnormal insulin production from pancreas etc. In order to combat these diseases, scientist have developed many novelties, such as drugs, to keep these malfunctioning cells at bay. 
Albeit, medicine as such, has not reached to a point where most of the diseases can be eleminated. Also, bacteria have the capability to evolve antibiotic resistantance. Therefore, it is essential to find new drugs and antibiotics.
To find better drugs for cancer for example, scientist have programs to calculate if the "drug" binds to the protein causing the disease. To make the drug more efficient, it is vital to know the exact structure of the protein that needs to targeted and also that the drug itself only or mostly binds to it (if it targets other proteins it can kill the organism). Computer scientists have come up with revolutionary softwares that predicts the structure of the protein and the exact binding of the drug to the protein. However, these algorithms within the software are only 70-80% accurate and thus making new drugs unreliable for therapeutic approach - further and more expensive experiments are needed to be done to verify the binding and the structure of the proteins.

<img src="https://github.com/tammekasra/A.I-in-Biomedicine/blob/main/folding.jfif" width="400">

Aritifical intelligence has helped to predict the folding of the protein and the binding of the drugs to 98% accuracy. This has given scientists a "boost" in drug discoveries making an augmented approaches in overall medicine. Although, 98% is a lot, but the 2% can still be detrimental and give "false" positive results. Therefor, it is still necessary to conduct further experiments in verifying the results. 

<img src="https://github.com/tammekasra/A.I-in-Biomedicine/blob/main/folding.png" width="400">
If this prediction can give a 100% probability, scientist can save money, time and other resources to make a research. As such, the idea of this project is to find a way to get the "extra" 2% accuracy in predicting protein folding and drug binding.

## Background

Which problems does your idea solve? How common or frequent is this problem? What is your personal motivation? Why is this topic important or interesting?
My own personal motivation on this topic is that I myself am I Biomedicine student whom would like to participate in cancer research labs.
If scientist could get hold on an algorithm that predicts the structure of a protein just from the DNA template itself, it would save a lot of time and money. Also, it could aid the prediction on how a cancer drug functions on a molecular level and therefor could also aid in understanding how a cancer cell is formed and how it evolves to be malicious - unstable growth and metastasis.
This problem in finding the accruate binding of a drug and the correct folding of a protein can take up to 5 to 10 years: using X-cristallography and different mouse models to visualize the binding of the drug and the structure of the targeted protein.


## How is it used?

Describe the process of using the solution. In what kind situations is the solution needed (environment, time, etc.)? Who are the users, what kinds of needs should be taken into account?

Firstly, a DNA sample should be taken from an individual (mouse, human etc) and perform NextGeneration Sequencing that gives the whole DNA strand of the individual. 
Secondly, a subset of a specific strand if found using various software. 
Thirdly the DNA strand is but in the algorithm where the protein folding is predicted from the gene.
Lastly, the folded protein simulation is used to find the appriopiate binding of a specific drug and is evaulated wheter it is a good candidate for theraupetic approach or not.

<img src="https://github.com/tammekasra/A.I-in-Biomedicine/blob/main/docking.png" width="400">

The intial procedure can use the following code below using python and data from ncbi.nlm.nih.gov

(https://github.com/aqlaboratory/proteinnet)






## Data sources and AI methods

If you need to use links, here's an example:
The data that is used in these softwares and either be optained using X-cristallography (if the protein is that yet charaterised) or through the website [Twitter API](https://www.ncbi.nlm.nih.gov/protein?cmd=retrieve)

<img src="https://github.com/tammekasra/A.I-in-Biomedicine/blob/main/x-cristallography.jfif" width="400">

It would be beneficial to optain many algorithms with atleast 98% accuracy. These algorithms could be used to combined to one another to make it even more accurate.


## Challenges

The main challenged facing this project is to find all the possible protein and that the algorithm indeed can solve all the protein foldings; if there are around 25,000 different proteins, therefor it can be possible that the algorithm can predict around 24,999 proteins structure yet that 1 protein can be forever be solved incorrectly or that a drug can actually be good and solve everything but can be undiscovered due to the 0,001% error.
Therefor, the biggest challenge is the assess and take into account even the smallest of errors.


## What next?

If this project could be done (it would however take a really long time to achieve this) it can solve most of the problems within the field of medicine.



## Acknowledgments
  https://www.nature.com/articles/d41586-019-01357-6
   https://github.com/aqlaboratory/proteinnet (code)
  https://www.nature.com/articles/d41586-020-03348-4
  

LICENSE to distribute this code - https://github.com/aqlaboratory/proteinnet/blob/master/LICENSE
