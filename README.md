# Learning from Label Relationships in Human Affect

Authors official PyTorch implementation of the **[Learning from Label Relationships in Human Affect](https://arxiv.org/pdf/2207.05577.pdf)**. If you use this code for your research, please [**cite**](#citation) our paper.

> **Learning from Label Relationships in Human Affect**<br>
> Niki Maria Foteinopoulou and Ioannis Patras<br>
> https://arxiv.org/abs/2207.05577 <br>
> ![summary](figs/summary.svg)
>
> **Abstract**: Human affect and mental state estimation in an automated manner, face a number of difficulties, including learning from labels with poor or no temporal resolution, learning from few datasets with little data (often due to confidentiality constraints) and, (very) long, in-the-wild videos. For these reasons, deep learning methodologies tend to overfit, that is, arrive at latent representations with poor generalisation performance on the final regression task. To overcome this, in this work, we introduce two complementary contributions. First, we introduce a novel relational loss for multilabel regression and ordinal problems that regularises learning and leads to better generalisation. The proposed loss uses label vector inter-relational information to learn better latent representations by aligning batch label distances to the distances in the latent feature space. Second, we utilise a two-stage attention architecture that estimates a target for each clip by using features from the neighbouring clips as temporal context. We evaluate the proposed methodology on both continuous affect and schizophrenia severity estimation problems, as there are methodological and contextual parallels between the two. Experimental results demonstrate that the proposed methodology outperforms the baselines that are trained using the supervised regression loss, as well as pre-training the network architecture with an unsupervised contrastive loss. In the domain of schizophrenia, the proposed methodology outperforms previous state-of-the-art by a large margin, achieving a PCC of up to 78% performance close to that of human experts (85%) and much higher than previous works (uplift of up to 40%). In the case of affect recognition, we outperform previous vision-based methods in terms of CCC on both the OMG and the AMIGOS datasets. Specifically for AMIGOS, we outperform previous SoTA CCC for both arousal and valence by 9% and 13% respectively, and in the OMG dataset we outperform previous vision works by up to 5% for both arousal and valence.


## Overview

![Overview](./figs/overview.svg)
<p alighn="center">
In a nutshell, the proposed architecture consists of two branches with shared weights, that incorporate two main components: 
a) a video-clip encoder employing a convolutional backbone network for frame-level feature extraction and 
b) a Transformer-based network leveraging the temporal relationships of the spatial features for clip-level feature extraction. 
The clip and context features produced by the aforementioned branches are passed to a context-based attention block and a regression head. 
The proposed method uses the context-based attention block to incorporate features from the two branches before passing 
them to the regression head. The bottom branch uses the proposed video-clip encoder to extract clip-level features 
from the input video clips, which subsequently feed the context-based attention block and are further used to construct 
the intra-batch similarity matrix for calculating the proposed relational loss. 
The goal of the proposed relational loss, as an additional auxiliary task to the main regression, is to obtain a more 
discriminative set of latent clip-level features, by aligning the label distances in the mini-batch to the latent 
feature distances. Finally, the upper branch uses the proposed video-clip encoder to extract clip-level features from 
the input video clips and a set of context clips from each of the input clips, which subsequently feed the regression 
head in order to infer the desired values and calculate the regression loss.
</p>


## Installation

We recommend installing the required packages using python's native virtual environment as follows:

```bash
$ python -m venv venv
$ source venv/bin/activate
(venv) $ pip install --upgrade pip
(venv) $ pip install -r requirements.txt
```

For using the aforementioned virtual environment in a Jupyter Notebook, you need to manually add the kernel as follows:

```bash
(venv) $ python -m ipykernel install --user --name=venv
```


## Citation

```bibtex
@inproceedings{foteinopoulou2022learning,
  title={Learning from Label Relationships in Human Affect},
  author={Foteinopoulou, Niki Maria and Patras, Ioannis},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={80--89},
  year={2022}
}

```



<!--Acknowledgement: This research was supported by the EU's Horizon 2020 programme H2020-951911 [AI4Media](https://www.ai4media.eu/) project.-->

