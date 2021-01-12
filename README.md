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
The intial procedure can use the following code below (https://github.com/aqlaboratory/proteinnet)
```
"""
TF parser for ProteinNet Records.
"""

__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

import tensorflow as tf

NUM_AAS = 20
NUM_DIMENSIONS = 3

def masking_matrix(mask, name=None):
    """ Constructs a masking matrix to zero out pairwise distances due to missing residues or padding. 
    Args:
        mask: 0/1 vector indicating whether a position should be masked (0) or not (1)
    Returns:
        A square matrix with all 1s except for rows and cols whose corresponding indices in mask are set to 0.
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH]
    """

    with tf.name_scope(name, 'masking_matrix', [mask]) as scope:
        mask = tf.convert_to_tensor(mask, name='mask')

        mask = tf.expand_dims(mask, 0)
        base = tf.ones([tf.size(mask), tf.size(mask)])
        matrix_mask = base * mask * tf.transpose(mask)

        return matrix_mask
        
def read_protein(filename_queue, max_length, num_evo_entries=21, name=None):
    """ Reads and parses a ProteinNet TF Record. 
        Primary sequences are mapped onto 20-dimensional one-hot vectors.
        Evolutionary sequences are mapped onto num_evo_entries-dimensional real-valued vectors.
        Secondary structures are mapped onto ints indicating one of 8 class labels.
        Tertiary coordinates are flattened so that there are 3 times as many coordinates as 
        residues.
        Evolutionary, secondary, and tertiary entries are optional.
    Args:
        filename_queue: TF queue for reading files
        max_length:     Maximum length of sequence (number of residues) [MAX_LENGTH]. Not a 
                        TF tensor and is thus a fixed value.
    Returns:
        id: string identifier of record
        one_hot_primary: AA sequence as one-hot vectors
        evolutionary: PSSM sequence as vectors
        secondary: DSSP sequence as int class labels
        tertiary: 3D coordinates of structure
        matrix_mask: Masking matrix to zero out pairwise distances in the masked regions
        pri_length: Length of amino acid sequence
        keep: True if primary length is less than or equal to max_length
    """

    with tf.name_scope(name, 'read_protein', []) as scope:
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        context, features = tf.parse_single_sequence_example(serialized_example,
                                context_features={'id': tf.FixedLenFeature((1,), tf.string)},
                                sequence_features={
                                    'primary':      tf.FixedLenSequenceFeature((1,),               tf.int64),
                                    'evolutionary': tf.FixedLenSequenceFeature((num_evo_entries,), tf.float32, allow_missing=True),
                                    'secondary':    tf.FixedLenSequenceFeature((1,),               tf.int64,   allow_missing=True),
                                    'tertiary':     tf.FixedLenSequenceFeature((NUM_DIMENSIONS,),  tf.float32, allow_missing=True),
                                    'mask':         tf.FixedLenSequenceFeature((1,),               tf.float32, allow_missing=True)})
        id_ = context['id'][0]
        primary =   tf.to_int32(features['primary'][:, 0])
        evolutionary =          features['evolutionary']
        secondary = tf.to_int32(features['secondary'][:, 0])
        tertiary =              features['tertiary']
        mask =                  features['mask'][:, 0]

        pri_length = tf.size(primary)
        keep = pri_length <= max_length

        one_hot_primary = tf.one_hot(primary, NUM_AAS)

        # Generate tertiary masking matrix--if mask is missing then assume all residues are present
        mask = tf.cond(tf.not_equal(tf.size(mask), 0), lambda: mask, lambda: tf.ones([pri_length]))
        ter_mask = masking_matrix(mask, name='ter_mask')        

        return id_, one_hot_primary, evolutionary, secondary, tertiary, ter_mask, pri_length, keep


main()
```


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
