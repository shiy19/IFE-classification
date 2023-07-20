## A Multi-Task Method for Immunofixation Electrophoresis Image Classification



### Quick Tour

This is a pytorch implementation for our paper.

To run the code To run the code, please make sure you have prepared your IFE data following the same structure as follows (you can also refer to the examplar data in this repository):

./data (the IFE images)

./label (the csv file that contains additional information)

We found that there are three type of labels for our dataset. 

The term "label" include nine classes, Non-M , κ , λ,  IgG-κ , IgG-λ  , IgA-κ ,IgA-λ , IgM-κ  , IgM-λ ,  respectively represented by number 0-8.

The other two types of label could easily derived from "label". "label1" include four classes, Neg, G, A, M, representing the co-location between ELP lanes and heavy chain lanes. "label2" include three classes, Neg,  κ , λ, representing the co-location between ELP lanes and mild chain lanes. For example, a sample from label "IgG-κ", its severe label and mild label are G and κ.



### Dependencies

The software dependencies are listed in dependencies.txt.



### Training and Evaluation

For training, you can cd directory "./code" and run:

```
python main.py
```

Training and evaluating would both be completed, and the state_dict of model will be saved.

Just for evaluating, cd directory "./code" and run:

```
python main.py --no_train
```

Model will load the state_dict file and evaluating on your test set.



### Datasets

Due to the privacy issue, we cannot distribute the original IFE dataset used in our paper. The performances of our method are listed in the paper.

Some synthetic images and their labels could be found in ./data and ./label