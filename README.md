This code is the implementation of the paper 
# 'Complementing Representation Deficiency in Few-shot Image Classification: A Meta-Learning Approach'.
Abstract：Few-shot learning is a challenging problem that has attracted more and more attention recently since abundant training samples are difficult to obtain in practical applications. Meta-learning has been proposed to address this issue, which focuses on quickly adapting a predictor as a base-learner to new tasks, given limited labeled samples. However, a critical challenge for meta-learning is the representation deficiency since it is hard to discover common information from a small number of training samples or even one, as is the representation of key features from such little information. As a result, a meta-learner cannot be trained well in a high-dimensional parameter space to generalize to new tasks. Existing methods mostly resort to extracting less expressive features so as to avoid the representation deficiency. Aiming at learning better representations, we propose a meta-learning approach with complemented representations network (MCRNet) for few-shot image classification. In particular, we embed a latent space, where latent codes are reconstructed with extra representation information to complement the representation deficiency. Furthermore, the latent space is established with variational inference, collaborating well with different base-learners, and can be extended to other models. Finally, our end-to-end framework achieves the state-of-the-art performance in image classification on three standard few-shot learning datasets.
![image](https://github.com/GuChenghs/MCRNet/blob/master/data/overview.png)

And this paper has been accepted by the 
# 25th International Conference on Pattern Recognition (ICPR2020).
The paper link is 
# http://arxiv.org/abs/2007.10778. 
If you are interested in my research or have any questions, please contact me at gucheng_hs@whut.edu.cn.
Good luck to you!

Citation
If you use this code for your research, please cite our paper.
Dependencies
Python 2.7+ (not tested on Python 3)
PyTorch 0.4.0+
qpth 0.0.11+
tqdm
Usage
Installation
Clone this repository:

git clone https://github.com/GuChenghs/MCRNet/
cd MCRNet
Download and decompress dataset files: miniImageNet (courtesy of Spyros Gidaris), tieredImageNet, FC100, CIFAR-FS

For each dataset loader, specify the path to the directory. For example, in data/mini_imagenet.py line 30:

_MINI_IMAGENET_DATASET_DIR = 'path/to/miniImageNet'
Meta-training
To train MCRNet-SVM on 5-way miniImageNet benchmark:
python train.py --gpu 0,1,2,3 --save-path "./experiments/miniImageNet_MetaOptNet_SVM" --train-shot 15 \
--head SVM --network ResNet --dataset miniImageNet --eps 0.1
As shown in Figure 2, of our paper, we can meta-train the embedding once with a high shot for all meta-testing shots. We don't need to meta-train with all possible meta-test shots unlike in Prototypical Networks.
You can experiment with varying base learners by changing '--head' argument to ProtoNet or Ridge. Also, you can change the backbone architecture to vanilla 4-layer conv net by setting '--network' argument to ProtoNet. For other arguments, please see MCRNet/train.py from lines 85 to 114.
To train MCRNet-SVM on 5-way tieredImageNet benchmark:
python train.py --gpu 0,1,2,3 --save-path "./experiments/tieredImageNet_MCRNet_SVM" --train-shot 10 \
--head SVM --network ResNet --dataset tieredImageNet
To train MCRNet-RR on 5-way CIFAR-FS benchmark:
python train.py --gpu 0 --save-path "./experiments/CIFAR_FS_MCRNet_RR" --train-shot 5 \
--head Ridge --network ResNet --dataset CIFAR_FS
To train MCRNet-RR on 5-way FC100 benchmark:
python train.py --gpu 0 --save-path "./experiments/FC100_MCRNet_RR" --train-shot 15 \
--head Ridge --network ResNet --dataset FC100
Meta-testing
To test MCRNet-SVM on 5-way miniImageNet 1-shot benchmark:
python test.py --gpu 0,1,2,3 --load ./experiments/miniImageNet_MCRNet_SVM/best_model.pth --episode 1000 \
--way 5 --shot 1 --query 15 --head SVM --network ResNet --dataset miniImageNet
Similarly, to test MCRNet-SVM on 5-way miniImageNet 5-shot benchmark:
python test.py --gpu 0,1,2,3 --load ./experiments/miniImageNet_MCRNet_SVM/best_model.pth --episode 1000 \
--way 5 --shot 5 --query 15 --head SVM --network ResNet --dataset miniImageNet
Acknowledgments
This code is based on the implementations of Prototypical Networks, Dynamic Few-Shot Visual Learning without Forgetting, and DropBlock.
