# Assignment 2 Report

### 3.1

Implemented K-Means class.



### 3.2

Plot of cost agains number of clusters k :  

![Elbow Curve for K-Means](plots/elbow%20curve.png)


**Parameters :**

**$k_{kmeans1}$ = 8**

+ This is the point in the cost vs component graph after which the relative change in the cost with the increasing k slows down.

**Clusters obtained :**

Cluster 7: 46 words  
drive, sing, rose, dive, smile, bear, bullet, shark, bury, bend, fly, face, climb, kneel, monkey, kiss, selfie, roof, catch, sleep, baseball, hollow, puppy, basket, empty, fish, slide, drink, arrest, bird, clock, angry, lazy, hang, snake, earth, tank, airplane, swim, cook, basketball, bicycle, recycle, shoe, sunny, truck

Cluster 6: 33 words  
deer, panda, helicopter, spider, giraffe, lizard, frog, cow, bed, starfish, sweater, sun, feet, pear, peacock, saturn, fruit, grape, ant, goldfish, spiderman, bee, tree, beetle, tent, tomato, dragonfly, parachute, butterfly, lantern, elephant, windmill, crocodile

Cluster 1: 19 words  
ape, sit, cat, eat, gym, rifle, pencil, dig, run, clap, pull, van, jacket, ambulance, camera, zip, car, pant, potato

Cluster 0: 24 words  
listen, flame, download, hard, fight, call, hit, far, cry, clean, sad, draw, pray, email, buy, burn, fire, close, scary, book, happy, loud, love, cut

Cluster 4: 12 words  
knock, exit, stairs, door, ladder, igloo, skate, enter, throne, key, wheel, walk

Cluster 2: 14 words  
needle, carrot, spoon, puppet, flute, scissor, finger, hammer, screwdriver, teaspoon, sword, knife, arrow, fork

Cluster 5: 28 words  
eraser, table, brick, mug, postcard, passport, microwave, notebook, microphone, bench, bucket, laptop, calendar, chair, mouse, oven, calculator, pillow, envelope, dustbin, television, loudspeaker, telephone, stove, toaster, keyboard, radio, suitcase

Cluster 3: 24 words  
fishing, grass, forest, brush, feather, lake, scream, paint, plant, knit, cigarette, boat, badminton, candle, toothbrush, tattoo, fingerprints, length, rain, pizza, rainy, toothpaste, comb, paintbrush



### 4.1

Implemented GMM class.


### 4.2

Plots for likelihood analysis against number of clusters : 

<h2 align="center">Log-Likelihood for Custom GMM</h2>

![Log Likelihood for GMM](plots/gmm_log_likelihood.png)

<h2 align="center">AIC-BIC Analysis for inbuilt sk-learn gmm</h2>

![AIC BIC Unnormalized - sklearn](plots/aic_bic_unnormalized.png)

<h2 align="center">AIC-BIC Analysis for normalized data using sk-learn</h2>

![AIC BIC Normalized - sklearn](plots/aic_bic_normalized.png)

<h2 align="center">AIC-BIC for custom implemented gmm class</h2>

![AIC BIC Custom implementation](plots/aic_bic_custom.png)

**Observations :** 

+ The grouping of words into clusters agrees with the semantic sense of the words. This suggests that the custom implemented gmm class works  fine.

+ The increasing log likelihood with the number of clusters k indicates the use of a higher value of k for clustering leads to better grouping.

+ The AIC and BIC criterion analysis for both the inbuilt sklearn class and the custom implementation are the same.
   + After performing PCA later, the AIC and BIC criterion give a better estimate of the optimum number of clusters.

**Parameters :**

**$k_{gmm1}$ = 2**

+ The AIC and BIC analysis is leading to this value of optimum cluster components. The reason for this inaccurate analysis is that AIC and BIC are information theoretic measures which don't actually report which k value leads to the best clustering, but rather they weigh the relative dependence of the two terms involved in theor expression and report that k which seems optimum as per that methodology.

+ Thus, while we know that $k_{gmm1}$ = 2 is not a good way to cluster words having a vast variety in semantics, yet it is what the analysis gives, for bith the inbuilt class as well for the self implemented class. 



### 5.1

Implemented PCA class.


### 5.2

The dataset after dimensionality reduction : 

![2D Visualization](plots/2d%20pca.png)
![3D Visualization](plots/3d%20pca.png)


### 5.3

**Observations from the 2D Plot :**

+ The axes correspond to the principal components. Principal component 1 corresponds to the largest eigen value, principal component 2 to the second largest and so on.

+ The X-Axis dimension seems to quantify the ***ability to action or move*** attribute of the entity. Hence, we observe that animals , such as bear, elephant, cow, panda are on the left side while hard, scary, happy, fire are clustered on the right side.

+ The Y-Axis dimension seems to quantify something similar to ***usability in context of verbi-ness*** of the entity, which is why we observe that teaspoon, door, notebook are clustered on the higher side while dig, run, eat are on the lower side.


**Observations from the 3D Plot :**

+ The Z-Axis dimension seems to segregate the words which correspond to **living beings**, which is why almost all the animals are clustered right at the top with the other two dimensions used to qualify other aspects.



**Parameters :**

**${k_2}$ = 4**

+ The above stated value seems visibly appropriate for clustering.




### 6.1 

Performed K-Means clustering using $k_{2}$.



### 6.2

Scree Plot :

![Scree Plot](plots/scree%20pca.png)


<br>

**As stated in the graph above, 95% of the variance is explained by the first 133 dimensions corresponding to the largest 133 eigen values.**

<br>


![Elbow Curve PCA](plots/kmeans%20pca.png)

+ For the reduced dataset, $k_{kmeans3}$ = 9 seems optimum for clustering based on the above elbow plot.


**Parameters :**

**$k_{kmeans3}$ = 9**



### 6.3 

Performed GMM using ${k_2}$ .


### 6.4

AIC BIC for Reduced Dataset :


![GMM PCA](plots/gmm%20pca.png)


**Observation :**

+ As per the AIC and BIC analysis, now we see the expexted ***V-Curve*** for the AIC - BIC plots.
+ The value corresponding to the  lowest value of the AIC curve is chosen as the  optimum k for clustering.

**Parameters :**

**$k_{gmm3}$ = 5**


### 7.1

### K-Means Cluster Analysis 

The optimal clusters as per my semantic analysis are around 4. However, after performing elbow method on original dataset, we obtain $k_{kmeans1}$ = 8, which forms clusters with high intra similarity but at the same time there are some words which should have belonged to a certain cluster but due to stricter classification are bound to another cluster.

After PCA, we obtain $k_{kmeans3}$ = 9 by elbow method, which again does not resolve the issue. 

Hence, $k_{2}$ = 4 seems as the optimal choice for clustering.

### 7.2

### GMM Cluster Analysis

As explained above, after analysing the words we find that the optimum k is 4. 
After applying PCA, we note that $k_{gmm3}$ = 5. Since this is the  optimum number of clusters required, as explained above, hence we find that $k_{gmm3}$ is optimal in case of GMM clustering.



### 7.3

### Compare K-Means and GMM

A study of the dataset leads to the conclusion that 4 clusters are what the words majorly fit into. 
On applying K-Means clustering, the elbow method does not indicate in any case that 4 clusters are what the optimum clusters are.
However, the analysis of gmm suggests that $k_{gmm3}$ = 5 is optimum.
While not exactly equal to k2=4, yet, the grouping produced here is the most well defined.


Thus, for the given dataset, $k_{gmm3}$ = 5 is optimum with GMM Clustering proving to be the better.



### 8 

![fig1](plots/Figure_1.png)
![fig1](plots/Figure_2.png)
![fig1](plots/Figure_3.png)
![fig1](plots/Figure_4.png)
![fig1](plots/Figure_5.png)
![fig1](plots/Figure_6.png)
![fig1](plots/Figure_7.png)
![fig1](plots/Figure_8.png)
![fig1](plots/Figure_9.png)
![fig1](plots/Figure_10.png)
![fig1](plots/Figure_11.png)


**Observation :**

+ $k_{best1}=4$

+ $k_{kbest2}=5$

+ The hierarchical clustering of the data represents different  levels of similarity between the words. Hence, words which are  very far in the lower levels, tend to get grouped under the same cluster as the  height of the levels increases.

+ Upon setting the number of clusters = 5, the nature of clusters developed after hierarchical clustering is quite similar to that of the GMM Clustering done for k = 5. 

+ For Euclidean distance measure, the ward linkage metric is analysed as the best for the task at hand here from the above plots.



### 9.1 

![scree pca](plots/scree%20spotify.png)

<br>

**As seen above, the 90% variance is explained by the first 12 components.**

<br>

Dimensionality reduction applied to ruduce the dimensions from 21 to 12.


<br>

**Parameters :**
<br>
**Distance metric = Manhattan**
<br>
**K=21**


### 9.2


**Observations :**


**Accuracy  = 0.1840**    
**Precision = 0.1519**  
**Recall    = 0.1806**  
**F1 score  = 0.1444**  


**Comparison :**

+ The KNN model gives nearly the same accuracy and performance metrics for the dimensionally reduced dataset as the original dataset.
The model however works signifacantly faster than the original which worked on all the dimensions.

+ Thus, there is a great improvement in terms of computational efficiency and only a slight deviation in the accuracy.


