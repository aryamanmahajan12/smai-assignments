# ARYAMAN MAHAJAN
# 2022102034


## 2.1 

Load the 'WineQT.csv' dataset into a dataframe using Pandas. For all the features, output the mean std dev, min and max attribute values.

Number of records  : 1143<br>
Number of features : 13

**Attribute  =  volatile acidity**<br>
Mean    : 0.5313385826771653<br>
Max     : 1.58<br>
Min     : 0.12<br>
Std Dev : 0.17955459612835617<br>  
**Attribute  =  citric acid**<br>
Mean    : 0.2683639545056868<br>
Max     : 1.0<br>
Min     : 0.0<br>
Std Dev : 0.19659979421574741<br>  
**Attribute  =  residual sugar**<br>
Mean    : 2.5321522309711284<br>
Max     : 15.5<br>
Min     : 0.9<br>
Std Dev : 1.355324197143589<br>    
**Attribute  =  chlorides**<br>
Mean    : 0.08693263342082239<br>
Max     : 0.611<br>
Min     : 0.012<br>
Std Dev : 0.04724665655215518 <br>   
**Attribute  =  free sulfur dioxide**<br>
Mean    : 15.615485564304462<br>
Max     : 68.0<br>
Min     : 1.0<br>
Std Dev : 10.246001115067605  <br>  
**Attribute  =  total sulfur dioxide**<br>
Mean    : 45.91469816272966<br>
Max     : 289.0<br>
Min     : 6.0<br>
Std Dev : 32.76778677994138  <br>  
**Attribute  =  density**<br>
Mean    : 0.9967304111986001<br>
Max     : 1.00369<br>
Min     : 0.99007<br>
Std Dev : 0.001924224834379527  <br>  
**Attribute  =  pH**<br>
Mean    : 3.3110148731408575<br>
Max     : 4.01<br>
Min     : 2.74<br>
Std Dev : 0.15659551281704315  <br>  
**Attribute  =  sulphates**<br>
Mean    : 0.6577077865266842<br>
Max     : 2.0<br>
Min     : 0.33<br>
Std Dev : 0.1703241580362606 <br>   
**Attribute  =  alcohol**<br>
Mean    : 10.442111402741325<br>
Max     : 14.9<br>
Min     : 8.4<br>
Std Dev : 1.0817221048833654 <br>   
**Attribute  =  quality**<br>
Mean    : 5.657042869641295<br>
Max     : 8<br>
Min     : 3<br>
Std Dev : 0.805471666920189  <br>  
**Attribute  =  Id**<br>
Mean    : 804.9693788276466<br>
Max     : 1597<br>
Min     : 0<br>
Std Dev : 463.79409851409224  <br>


## 2.2
### Distribution of the label - 'quality' across the dataset : 

![label_distribution](plots/label_distribution.png)




## 2.3

+ Use dropna() function from Pandas library in its default setting to drop all the records from the dataframe with any NaN value(missing value).

+ Use inplace=True command in dropna() function to directly modify the original dataset.

+ Apply MinMaxScaler() to the cleaned dataframe that applies min max normalization to the entire data. 
    This scaler transforms features by scaling them to a specified range, here [0, 1].<br>
    The formula for normalization is :
    $$x_{\text{scaled}} = \frac{x - \text{min}}{\text{max} - \text{min}}$$

+ Convert the normalized data back to another dataframe.

+ Apply StandardScaler() to the normalized dataframe that standardizes the distribution of samples across the dataset. The standardization process converts the data to zero mean and unit variance data.

+ For each feature (column) in the dataset, the mean and standard deviation are computed as follows:<br>

    $$\text{mean} = \frac{1}{n} \sum_{i=1}^{n}$$

    $$\text{std dev} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \text{mean})^2}$$
+ The standardization applied is :
    $$z = \frac{x - \text{mean}}{\text{std dev}}$$
