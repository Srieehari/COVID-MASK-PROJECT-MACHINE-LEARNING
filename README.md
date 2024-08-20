**Project's Title**
Impact of Hidden Layer Depth on Loss and Computational Efficiency in 
Facial Image Classification Using Neural Networks

**Introduction**

Pandemics have always been apparent in civilization, wreaking havoc on existing societal structures. From the Black Death to the Flu pandemic, millions of lives have been lost due to pandemics. More recently, the COVID-19 pandemic raged on, affecting not just human health but the global economy and food systems. First discovered in Wuhan, China, in November 2019, the virus spread easily across the globe because “public awareness on the virus prevention” was very little (Hao et al.). My own uncle succumbed to the virus and sadly passed away due to the lack of preventative measures placed in his home country. COVID-19 mainly spreads through human-to-human transmission making it highly infective and especially deadly to humans with weak immune systems. To limit the spread, governmental organizations enforced protective measures, including mandated lockdowns of public areas and maintaining a six-foot distance between individuals in closed areas. However, among the various measures imposed by governmental organizations, the mask mandate has proven to be one of the most effective in controlling the spread of the virus. The principle behind mandating face masks is straightforward: since Covid-19 primarily spreads through respiratory droplets, wearing a face mask acts as a barrier, preventing these droplets from entering one's nose, throat, and lungs. Although the idea is simple, forcing millions of people around the world to wear a face mask poses an incredible challenge for governmental organizations. As a result, when observing a general population, it becomes evident that many people wear face masks properly, while others wear them improperly, and there are also some individuals who do not wear face masks at all. Generally, this discrepancy is ignored by humans; however, by utilizing computer vision algorithms, the detection of whether a mask is worn can be done autonomously. This project aims to investigate the extent to which modifying layers in a neural network can affect the performance of a computer vision algorithm, specifically their loss and analysis speed, in classifying images of humans wearing face masks, whether they are wearing them properly, improperly, or not at all. To investigate the relationship, a computer vision algorithm was programmed to analyze various images of humans wearing masks properly, improperly, and not at all collected from a public dataset. The number of layers was altered, and speed and loss were analyzed. Explanations and results were obtained and discussed. This project could prove helpful in all public settings. Although the Covid-19 pandemic is over, wearing masks can be highly effective in reducing the spread of many different diseases. Through the investigation of optimizing machine learning mask detection models, the enforcement of more effective mask mandates becomes possible. By implementing this technology in highly populated public areas, the chances of viral transmission can be reduced by detecting individuals not wearing masks. Hopefully, this will contribute to the overall public health and prevent any more of my loved ones from getting ill.

The models I created had a varying amount of layers:

1. 1 Hidden layer. 
2. 3 Hidden layers 
3. 7 Hidden layers 
4. 10 Hidden layers 
5. 12 Hidden layers 

**Results**
	**Training and Testing Speeds**
As the number of layers increases, there is a significant increase in the time taken to train the models. There is a greater increase in the training phase compared to the testing phase. In other words, while the training time experiences a significant increase, the testing time shows a slower rate of increase with each additional layer. 
**	Training and Testing Loss ** 
There is an inverse relationship between the number of layers and loss. This means that as the layer count increases the training loss exhibits a significant decrease. The testing loss also experiences a decrease; however, this rate is much slower. 
**Relationship between Loss and Layers ** 
The most significant reduction in loss occurs between models with 1 layer and 3 layers. However, as the amount of layers increases, the decrease in loss becomes more gradual. This is apparent in the models with 10 and 12 layers as their values are very close together.
**	Relationship between Time and Layers **
On the other hand, a direct relationship is observed between the number of layers and the time required for model training. The greatest increase in training time occurs between models with 1 layer and 3 layers. However, as the layer count continues to rise, the increase in training time becomes more gradual.

**Interpretation of Results**

Training Set:
The observed inverse correlation between layer count and training loss shows that the addition of extra layers improves the model’s representational capacity—the ability to learn from a dataset. Lower layers of the model recognize patterns from simple features while higher layers are able to detect more complex patterns. With more layers a model is able to capture more patterns in the dataset. This can clearly be seen in the reduction of training loss between the models with 3 and 7 layers. 

However, the gradual decrease in loss observed in the models with 10 and 12 layers suggests that there is a point of diminishing returns. One reason for this could be that the model is beginning to memorize training data, leading to a plateau in training loss. This issue is called overfitting. However, since the loss is still decreasing, overfitting did not completely occur in this experiment. Regardless, the plateau emphasizes how important it is to optimally choose the number of layers to include in a CNN model. If this is not done, at a certain point more computational power will be utilized for a loss that is not much lower than a model trained with fewer layers.

The increase in training time is associated with the computational burden associated with adding layers. Each layer requires more computations leading to an increase in total training time. This trade-off between model complexity and training time emphasizes the importance of considering computational constraints when optimizing a model. 

Testing Set:
In the testing set, the inverse relationship between layer count and validation loss indicates that there is a benefit in increasing layer count beyond just the training set. By increasing the number of layers, the machine learning is more able to accurately make predictions on data it has never been tested on because it is able to detect more patterns.

However, the overfitting challenge is still apparent in the testing set as there is clear plateauing of the validation loss with the models with 10 and 12 hidden layers. 

The parallel increase in testing time with layer count demonstrates the computational demand required during the testing phase. Although the testing time is significantly lower than the training time this is due to the fact that the model utilizes fewer images and as a result takes less time to output a result.

The observed trends in both the training and validation sets highlight the need for a balanced model that not only minimizes loss but also operates within a reasonable time. 



**Further Research**
Overfitting
In this investigation, the model used a maximum of 12 hidden layers. Even at this amount, the loss began to plateau and decreased at a lower rate. As a result, an interesting investigation point would be to experiment with what number of layers the model would begin showing signs of overfitting. The initial number of hidden layers can begin with 12 and increase from that point.

Video Analysis
In this investigation, the dataset used was comprised of face mask images. However, a more practical utilization of face mask detection would be for videos because they provide more real-time updates. A new investigation perspective would be to determine how changing hidden layer count in video recognition machine learning models can affect training time and loss.

Conclusion
In this paper, the effects of changing the number of hidden layers on a CNN’s training loss, testing loss, training time, and testing time were analyzed. An explanation for the patterns observed was also provided. The results show that increasing the number of hidden layers in a CNN reduces the model’s training and testing loss and increases the training and testing time. 

Increasing the number of hidden layers has benefits up to a certain point. It is true that the training loss is much lower. However, after a certain number of layers, the loss begins to plateau. There is not much difference, and eventually, overfitting will occur. At this stage, the model’s loss will increase. On the other hand, the model's training time increases as the number of hidden layers increases. 

Overall, the most optimal number of layers is the one that has minimum loss while also not having extreme training time. For this specific experiment, the model with 7 hidden layers offers the most optimal number of layers as it balances all 4 variables. 

This paper can have numerous benefits to computer vision model developers intending to develop the most optimal machine learning algorithms. This in turn can save computational resources and achieve better performance. Additionally, this paper extends its significance to the societal well-being of humans around the world. It shows the implications of how computer vision can be used for the improvement of the health and safety of numerous communities around the world by ensuring stricter mask protocols are enforced.  


**Dataset:**

The dataset I used was found on Kaggle. The creators name was Larxel. Since the file size of the dataset was too large even after compressed it is much easier to dowload it from this link.

https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

**Using the code**
To build this project download the file and the datasets. There is only one section of the code that needs to be altered and it is the amount of layers you want to include. The amount can be increased or decreased depending on the user preference.

**Here is a list of research articles I read and interpreted to understand how a CNN works. Reading these can help you not only on this project but for any ML project.**

Abiodun, Oludare Isaac, et al. “State-of-the-Art in Artificial Neural Network Applications: A Survey.” Heliyon, vol. 4, no. 11, 2018, https://doi.org/10.1016/j.heliyon.2018.e00938. Accessed 4 Aug. 2023.

Hao YJ, Wang YL, Wang MY, Zhou L, Shi JY, Cao JM, Wang DP. The origins of COVID-19 pandemic: A brief overview. Transbound Emerg Dis. 2022 Nov;69(6):3181-3197. doi: 10.1111/tbed.14732. Epub 2022 Oct 20. PMID: 36218169; PMCID: PMC9874793. Accessed 4 Aug. 2023.

Huang, Jonathan, et al. “Speed/Accuracy Trade-Offs for Modern Convolutional Object Detectors.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, https://doi.org/10.1109/cvpr.2017.351. Accessed 4 Aug. 2023.
Kang, Laisong. “Wave Monitoring Based on Improved Convolution Neural Network.” Journal of Coastal Research, 2019, pp. 186–90. JSTOR, https://www.jstor.org/stable/26853931. Accessed 4 Aug. 2023.

Rand, Lindsay, et al. “Computer Vision.” Emerging Technologies and Trade Controls: A Sectoral Composition Approach, Center for International & Security Studies, U. Maryland, 2020, pp. 68–86. JSTOR, http://www.jstor.org/stable/resrep26934.9. Accessed 2 Aug. 2023.

Verpoort, Philipp C., et al. “Archetypal Landscapes for Deep Neural Networks.” Proceedings of the National Academy of Sciences of the United States of America, vol. 117, no. 36, 2020, pp. 21857–64. JSTOR, https://www.jstor.org/stable/26969086. Accessed 2 Aug. 2023.

Withorne, Jamie. “Machine Learning Precedents.” Machine Learning Applications in Nonproliferation: Assessing Algorithmic Tools for Strengthening Strategic Trade Controls, James Martin Center for Nonproliferation Studies (CNS), 2020, pp. 9–12. JSTOR, http://www.jstor.org/stable/resrep26358.6. Accessed 2 Aug. 2023.


Overall I learned a great deal about CNN from this project. I learned how to preprocess images and alter them to be in a format that can be understood by the ML model. I also learned the basic structure of Neural Network. Using the knowledge I learned from this project I hope to develop more effective ML models that are more practical.
