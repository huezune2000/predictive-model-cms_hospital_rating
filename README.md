# CMS Hospital Rating¶

	Fun fact: The probability of picking a 5-star rating hospital in the USA that is from Texas is 0.35%.
	Fun fact: The probability of picking a 1-star rating hospital in the USA that is from New York is 1.1%.



CMS Hospital Rating
Fun fact: The probability of picking a 5-star rating hospital in the USA that is from Texas is 0.35%.
Fun fact: The probability of picking a 1-star rating hospital in the USA that is from New York is 1.1%.
It is every hospital's dream to have a 5-star rating as it signifies the level of patient-care quality provided. The rating is one of many ways to showcase quality recognition and reputation. The dataset by Centers of Medicare and Medicaid, was downloaded on Kaggle dataset. This data can help us compare the quality of care among hospitals in the USA. This is an interesting project as the data can be used for predictive modeling using Python. The result of the machine learning algorithms can have implications:
Hospitals - better resource-allocation strategy
Insurance - improve overall policies/practices and network coverage strategy
Patients - better understanding of their healthcare facilities and makes informed decision
The objective of this project is to predict the hospital's rating, thus, the target variable or the y-variable is "Hospital overall rating".
The project included comprehensive exploratory data analysis, data cleansing, and data visualization. Multiclass predictive models using machine learning from sci-kit learn libraries included K-nearest neighbor, Support Vector Machines, and Random Forest. The end result included the comparison of these models' performance.
Importing libraries
In [248]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
Opening and reading downloaded dataset
In [249]:
data = pd.read_excel("Hospital General Information.xlsx")
In [250]:
data.head()
Out[250]:
Provider ID	Hospital Name	Address	City	State	ZIP Code	County Name	Phone Number	Hospital Type	Hospital Ownership	...	Readmission national comparison	Readmission national comparison footnote	Patient experience national comparison	Patient experience national comparison footnote	Effectiveness of care national comparison	Effectiveness of care national comparison footnote	Timeliness of care national comparison	Timeliness of care national comparison footnote	Efficient use of medical imaging national comparison	Efficient use of medical imaging national comparison footnote
0	10001	SOUTHEAST ALABAMA MEDICAL CENTER	1108 ROSS CLARK CIRCLE	DOTHAN	AL	36301	HOUSTON	3347938701	Acute Care Hospitals	Government - Hospital District or Authority	...	Same as the national average	NaN	Below the national average	NaN	Same as the national average	NaN	Same as the national average	NaN	Same as the national average	NaN
1	10005	MARSHALL MEDICAL CENTER SOUTH	2505 U S HIGHWAY 431 NORTH	BOAZ	AL	35957	MARSHALL	2565938310	Acute Care Hospitals	Government - Hospital District or Authority	...	Above the national average	NaN	Same as the national average	NaN	Same as the national average	NaN	Above the national average	NaN	Below the national average	NaN
2	10006	ELIZA COFFEE MEMORIAL HOSPITAL	205 MARENGO STREET	FLORENCE	AL	35631	LAUDERDALE	2567688400	Acute Care Hospitals	Government - Hospital District or Authority	...	Same as the national average	NaN	Below the national average	NaN	Same as the national average	NaN	Above the national average	NaN	Same as the national average	NaN
3	10007	MIZELL MEMORIAL HOSPITAL	702 N MAIN ST	OPP	AL	36467	COVINGTON	3344933541	Acute Care Hospitals	Voluntary non-profit - Private	...	Below the national average	NaN	Same as the national average	NaN	Below the national average	NaN	Above the national average	NaN	Not Available	Results are not available for this reporting p...
4	10008	CRENSHAW COMMUNITY HOSPITAL	101 HOSPITAL CIRCLE	LUVERNE	AL	36049	CRENSHAW	3343353374	Acute Care Hospitals	Proprietary	...	Same as the national average	NaN	Not Available	Results are not available for this reporting p...	Same as the national average	NaN	Above the national average	NaN	Not Available	Results are not available for this reporting p...
5 rows × 28 columns

Checking out the info on dataset to see what am I dealing with. Noted that features ending with *footnote have low non-null values. Almost all are objects except for provider ID, ZIP code, and phone numbers are numbers, which makes sense. The dataset is categorical-heavy. This means categorical graphs will be used frequently during the exploratory data analysis.
In [251]:
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4812 entries, 0 to 4811
Data columns (total 28 columns):
 #   Column                                                         Non-Null Count  Dtype 
---  ------                                                         --------------  ----- 
 0   Provider ID                                                    4812 non-null   int64 
 1   Hospital Name                                                  4812 non-null   object
 2   Address                                                        4812 non-null   object
 3   City                                                           4812 non-null   object
 4   State                                                          4812 non-null   object
 5   ZIP Code                                                       4812 non-null   int64 
 6   County Name                                                    4797 non-null   object
 7   Phone Number                                                   4812 non-null   int64 
 8   Hospital Type                                                  4812 non-null   object
 9   Hospital Ownership                                             4812 non-null   object
 10  Emergency Services                                             4812 non-null   object
 11  Meets criteria for meaningful use of EHRs                      4669 non-null   object
 12  Hospital overall rating                                        4812 non-null   object
 13  Hospital overall rating footnote                               1398 non-null   object
 14  Mortality national comparison                                  4812 non-null   object
 15  Mortality national comparison footnote                         1352 non-null   object
 16  Safety of care national comparison                             4812 non-null   object
 17  Safety of care national comparison footnote                    2168 non-null   object
 18  Readmission national comparison                                4812 non-null   object
 19  Readmission national comparison footnote                       1017 non-null   object
 20  Patient experience national comparison                         4812 non-null   object
 21  Patient experience national comparison footnote                1369 non-null   object
 22  Effectiveness of care national comparison                      4812 non-null   object
 23  Effectiveness of care national comparison footnote             1202 non-null   object
 24  Timeliness of care national comparison                         4812 non-null   object
 25  Timeliness of care national comparison footnote                1266 non-null   object
 26  Efficient use of medical imaging national comparison           4812 non-null   object
 27  Efficient use of medical imaging national comparison footnote  2033 non-null   object
dtypes: int64(3), object(25)
memory usage: 1.0+ MB
Checking out the total null counts helps me decide to drop them completely or replace them with relevant values.
In [252]:
data.isnull().sum()
Out[252]:
Provider ID                                                         0
Hospital Name                                                       0
Address                                                             0
City                                                                0
State                                                               0
ZIP Code                                                            0
County Name                                                        15
Phone Number                                                        0
Hospital Type                                                       0
Hospital Ownership                                                  0
Emergency Services                                                  0
Meets criteria for meaningful use of EHRs                         143
Hospital overall rating                                             0
Hospital overall rating footnote                                 3414
Mortality national comparison                                       0
Mortality national comparison footnote                           3460
Safety of care national comparison                                  0
Safety of care national comparison footnote                      2644
Readmission national comparison                                     0
Readmission national comparison footnote                         3795
Patient experience national comparison                              0
Patient experience national comparison footnote                  3443
Effectiveness of care national comparison                           0
Effectiveness of care national comparison footnote               3610
Timeliness of care national comparison                              0
Timeliness of care national comparison footnote                  3546
Efficient use of medical imaging national comparison                0
Efficient use of medical imaging national comparison footnote    2779
dtype: int64
Diving in to see what is in *footnote - Mostly description, not very informational in our work here.
In [253]:
data["Hospital overall rating footnote"].value_counts()
Out[253]:
There are too few measures or measure groups reported to calculate a star rating or measure group score                                        995
Data are shown only for hospitals that participate in the Inpatient Quality Reporting (IQR) and Outpatient Quality Reporting (OQR) programs    159
This hospital�s star rating only includes data reported on inpatient services                                                                  153
Data suppressed by CMS for one or more quarters                                                                                                 49
Results are not available for this reporting period                                                                                             42
Name: Hospital overall rating footnote, dtype: int64
I have decided to just drop those *footnote features since the information is not relevant to this project.
In [254]:
data = data.drop(["Hospital overall rating footnote", "Mortality national comparison footnote", 
          "Safety of care national comparison footnote", "Readmission national comparison footnote",
         "Readmission national comparison footnote", "Patient experience national comparison footnote",
         "Effectiveness of care national comparison footnote", "Timeliness of care national comparison footnote",
         "Efficient use of medical imaging national comparison footnote"], axis=1)
Rechecking what is left - The length of the dataset is ~4800. I have decided to just drop the 15 null values on county name. Need to look into "Meets criteria for meaningful use of EHRs" and decide if dropping 143 null values will make sense.
In [255]:
data.isnull().sum()
Out[255]:
Provider ID                                               0
Hospital Name                                             0
Address                                                   0
City                                                      0
State                                                     0
ZIP Code                                                  0
County Name                                              15
Phone Number                                              0
Hospital Type                                             0
Hospital Ownership                                        0
Emergency Services                                        0
Meets criteria for meaningful use of EHRs               143
Hospital overall rating                                   0
Mortality national comparison                             0
Safety of care national comparison                        0
Readmission national comparison                           0
Patient experience national comparison                    0
Effectiveness of care national comparison                 0
Timeliness of care national comparison                    0
Efficient use of medical imaging national comparison      0
dtype: int64
In [256]:
data["Meets criteria for meaningful use of EHRs"].value_counts()
Out[256]:
Y                4668
Not Available       1
Name: Meets criteria for meaningful use of EHRs, dtype: int64
Almost 90% of the data for feature "Meets criteria for meaningful use of EHR" is Yes and 1 is Not Available. I will just drop this feature since the data is not too helpful.
In [257]:
data = data.dropna()
data.isnull().sum()
Out[257]:
Provider ID                                             0
Hospital Name                                           0
Address                                                 0
City                                                    0
State                                                   0
ZIP Code                                                0
County Name                                             0
Phone Number                                            0
Hospital Type                                           0
Hospital Ownership                                      0
Emergency Services                                      0
Meets criteria for meaningful use of EHRs               0
Hospital overall rating                                 0
Mortality national comparison                           0
Safety of care national comparison                      0
Readmission national comparison                         0
Patient experience national comparison                  0
Effectiveness of care national comparison               0
Timeliness of care national comparison                  0
Efficient use of medical imaging national comparison    0
dtype: int64
Now I have a non-null data to work with. The exploratory data analysis begins here. The features listed in the top line will be used as my roadmap for exploration. First, I will do a basic descriptive analysis.
In [258]:
data.describe()
Out[258]:
Provider ID	ZIP Code	Phone Number
count	4655.000000	4655.000000	4.655000e+03
mean	266722.448335	53780.188614	5.849839e+09
std	154614.918602	26804.881596	2.328346e+09
min	10001.000000	603.000000	9.369338e+08
25%	140194.000000	32995.000000	4.017335e+09
50%	260011.000000	55066.000000	6.053844e+09
75%	390179.500000	75818.000000	7.877360e+09
max	670114.000000	99901.000000	9.898943e+09
The descriptive analysis does not help much. The ZIP code will provide us some geographical information. I suspect Provider ID is a unique feature (primary key). Phone numbers don't help much. Planned to drop Provider ID if this feature is used as unique identifier, which is not helpful in the project. Also will drop phone numbers.
In [259]:
data["Provider ID"].nunique()
Out[259]:
4655
My dropping list now contains Provider ID, Phone Number, and Meets criteria for meaningful use of EHRs. Let's check county name if it is a useful feature.
In [260]:
data["County Name"].value_counts()
Out[260]:
LOS ANGELES          79
COOK                 51
JEFFERSON            46
WASHINGTON           43
MONTGOMERY           41
                     ..
CAYUGA                1
LOS ALAMOS            1
MATANUSKA SUSITNA     1
LYNCHBURG CITY        1
FOSTER                1
Name: County Name, Length: 1547, dtype: int64
The County Name feature has length of ~1500. It is not feasible to keep this feature in our machine learning process later. Moreover, the ZIP code should be enough to provide us geographical information. Planned to drop this feature as well.
In [261]:
data = data.drop(["County Name", "Meets criteria for meaningful use of EHRs", "Provider ID", "Phone Number"], axis=1)
In [262]:
data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4655 entries, 0 to 4803
Data columns (total 16 columns):
 #   Column                                                Non-Null Count  Dtype 
---  ------                                                --------------  ----- 
 0   Hospital Name                                         4655 non-null   object
 1   Address                                               4655 non-null   object
 2   City                                                  4655 non-null   object
 3   State                                                 4655 non-null   object
 4   ZIP Code                                              4655 non-null   int64 
 5   Hospital Type                                         4655 non-null   object
 6   Hospital Ownership                                    4655 non-null   object
 7   Emergency Services                                    4655 non-null   object
 8   Hospital overall rating                               4655 non-null   object
 9   Mortality national comparison                         4655 non-null   object
 10  Safety of care national comparison                    4655 non-null   object
 11  Readmission national comparison                       4655 non-null   object
 12  Patient experience national comparison                4655 non-null   object
 13  Effectiveness of care national comparison             4655 non-null   object
 14  Timeliness of care national comparison                4655 non-null   object
 15  Efficient use of medical imaging national comparison  4655 non-null   object
dtypes: int64(1), object(15)
memory usage: 618.2+ KB
There are a few more features to consider dropping before moving to the core areas - Hospital Name and Address. I'm not too sure if these features are useful in this project.
In [263]:
data["Hospital Name"].value_counts()
Out[263]:
MEMORIAL HOSPITAL                                15
COMMUNITY MEMORIAL HOSPITAL                       7
GOOD SAMARITAN HOSPITAL                           7
ST JOSEPH HOSPITAL                                7
ST FRANCIS HOSPITAL                               6
                                                 ..
ATCHISON HOSPITAL                                 1
PIONEER HEALTH SERVICES OF PATRICK COUNTY,INC     1
SAINT FRANCIS BARTLETT MEDICAL CENTER             1
FLOWER HOSPITAL                                   1
MEMORIAL HOSPITAL OF TAMPA                        1
Name: Hospital Name, Length: 4473, dtype: int64
In [264]:
data["Address"].value_counts()
Out[264]:
100 HOSPITAL DRIVE                           7
100 MEDICAL CENTER DRIVE                     4
200 HOSPITAL DRIVE                           4
ONE HOSPITAL DRIVE                           3
100 HOSPITAL ROAD                            2
                                            ..
1206 GORDON DUCKWORTH DRIVE                  1
845 PARKSIDE ST                              1
1800 UNIVERSITY BOULEVARD                    1
1601 WATSON BOULEVARD                        1
AVE. PEDRO ALBIZU CAMPOS URB. LA HACIENDA    1
Name: Address, Length: 4628, dtype: int64
In [265]:
data = data.drop(["Address", "Hospital Name"], axis=1)
Comprehensive Exploratory Data Analysis
In [266]:
data["Hospital Type"].value_counts()
Out[266]:
Acute Care Hospitals         3256
Critical Access Hospitals    1308
Childrens                      91
Name: Hospital Type, dtype: int64
The majority of the hospitals in the USA are acute care hospitals, followed by critical access, and children.
In [267]:
plt.figure(figsize=(12,5))
sns.set_context("paper", font_scale=1.5)
sns.countplot(x=data["Hospital Type"], data=data, color="red", alpha=0.4)
Out[267]:
<AxesSubplot:xlabel='Hospital Type', ylabel='count'>

In [268]:
data["Emergency Services"].value_counts()
Out[268]:
Yes    4398
No      257
Name: Emergency Services, dtype: int64
The majority of the acute care hospitals and critical access have some sort of Emergency Services, unlike children's hospital. This makes sense as children's hospital is a specialty, usually affiliated with large acute care hospital in health systems.
In [269]:
plt.figure(figsize=(12,7))
sns.countplot(x="Hospital Type", data=data, hue="Emergency Services", palette="ocean_r")
Out[269]:
<AxesSubplot:xlabel='Hospital Type', ylabel='count'>

In [270]:
plt.figure(figsize=(9,5))
sns.set_context("paper", font_scale=1.5)
sns.countplot(x="Emergency Services", data=data, color="blue", alpha=0.4)
Out[270]:
<AxesSubplot:xlabel='Emergency Services', ylabel='count'>

In [271]:
data["Hospital Ownership"].value_counts()
Out[271]:
Voluntary non-profit - Private                 2003
Proprietary                                     753
Government - Hospital District or Authority     544
Voluntary non-profit - Other                    456
Government - Local                              392
Voluntary non-profit - Church                   341
Government - State                               60
Physician                                        55
Government - Federal                             42
Tribal                                            9
Name: Hospital Ownership, dtype: int64
The majority of the hospitals in the USA is private non-profit, government and faith-based. This is followed by physician-owned hospitals and tribal-owned bottoms the list. It is probably due to the total population of the tribes is relatively low compared to the non-tribes. The supply and demand laws in healthcare applies here.
In [272]:
plt.figure(figsize=(12, 7))
sns.set_context("paper", font_scale=1.5)
order = data["Hospital Ownership"].value_counts().sort_values(ascending=False).index
sns.countplot(y="Hospital Ownership", data=data, color="green", alpha=0.4, order=order)
Out[272]:
<AxesSubplot:xlabel='count', ylabel='Hospital Ownership'>

In [273]:
data["Hospital Ownership"].value_counts(normalize=True)*100
Out[273]:
Voluntary non-profit - Private                 43.029001
Proprietary                                    16.176155
Government - Hospital District or Authority    11.686359
Voluntary non-profit - Other                    9.795918
Government - Local                              8.421053
Voluntary non-profit - Church                   7.325456
Government - State                              1.288937
Physician                                       1.181525
Government - Federal                            0.902256
Tribal                                          0.193340
Name: Hospital Ownership, dtype: float64
In [274]:
data["Hospital overall rating"].value_counts()
Out[274]:
3                1751
Not Available    1113
4                 931
2                 672
1                 107
5                  81
Name: Hospital overall rating, dtype: int64
Planned to drop the Not Available even though its count is significant. Replacing it with the mean value does not provide us an accurate picture, which is representative of the hospital rating in this country.
In [275]:
data = data.drop(data[data["Hospital overall rating"] == "Not Available"].index)
In [276]:
data["Hospital overall rating"].value_counts(normalize=True)*100
Out[276]:
3    49.435347
4    26.284585
2    18.972332
1     3.020892
5     2.286844
Name: Hospital overall rating, dtype: float64
As expected, the majority of the hospitals in the USA falls under the average rating of 3 (49%). The extreme rating 1 and 5, with 5 being the best rating, have the lowest percentages. This is expected as it shows a very small number of hospitals (3%) are actually under-performed and it is very difficult for hospitals to obtain 5-star rating (only 2%) as performance is based on CMS value-based programs and so forth. At this stage, it is hard to tell if hospitals have resources constraints as the barrier versus CMS requirements are too high.
In [277]:
plt.figure(figsize=(15,7))
sns.set_context("poster", font_scale=1)
sns.countplot(x=data["Hospital overall rating"], palette="plasma")
Out[277]:
<AxesSubplot:xlabel='Hospital overall rating', ylabel='count'>

Based on the graph above, it is safe to assume that the majority of the hospitals in the USA lean toward average to higher rating (average performers and over-achievers) with smaller percentage of them on the low-end side. In general, hospitals are performing better than worse based on CMS requirements with only abur less than a quarter ~21% are under=performing. This does not reflect the true picture of the current hospital performance; over-time tracking is required to see if hospitals are improving or declining.
Continuing analysis - will drop Not Available counts in all features.
In [278]:
data["Mortality national comparison"].value_counts()
Out[278]:
Same as the national average    2575
Above the national average       400
Below the national average       340
Not Available                    227
Name: Mortality national comparison, dtype: int64
In [279]:
data = data.drop(data[data["Mortality national comparison"] == "Not Available"].index)
In [280]:
data["Mortality national comparison"].value_counts(normalize=True)*100
Out[280]:
Same as the national average    77.677225
Above the national average      12.066365
Below the national average      10.256410
Name: Mortality national comparison, dtype: float64
In [281]:
plt.figure(figsize=(15,7))
sns.set_context("paper", font_scale=1.5)
mortality_order = data["Mortality national comparison"].value_counts().sort_values(ascending=False).index
sns.countplot(x=data["Mortality national comparison"], color="pink", order=mortality_order)
Out[281]:
<AxesSubplot:xlabel='Mortality national comparison', ylabel='count'>

In [282]:
data["Safety of care national comparison"].value_counts()
Out[282]:
Same as the national average    1109
Not Available                    781
Above the national average       770
Below the national average       655
Name: Safety of care national comparison, dtype: int64
In [283]:
data = data.drop(data[data["Safety of care national comparison"] == "Not Available"].index)
In [284]:
data["Safety of care national comparison"].value_counts(normalize=True)*100
Out[284]:
Same as the national average    43.764799
Above the national average      30.386740
Below the national average      25.848461
Name: Safety of care national comparison, dtype: float64
In [285]:
plt.figure(figsize=(15,7))
sns.countplot(x=data["Safety of care national comparison"], color="orange")
Out[285]:
<AxesSubplot:xlabel='Safety of care national comparison', ylabel='count'>

In [286]:
data["Readmission national comparison"].value_counts()
Out[286]:
Same as the national average    961
Below the national average      807
Above the national average      766
Name: Readmission national comparison, dtype: int64
In [287]:
plt.figure(figsize=(15,7))
sns.countplot(x=data["Readmission national comparison"], color="grey")
Out[287]:
<AxesSubplot:xlabel='Readmission national comparison', ylabel='count'>

In [288]:
data["Patient experience national comparison"].value_counts()
Out[288]:
Below the national average      986
Same as the national average    837
Above the national average      703
Not Available                     8
Name: Patient experience national comparison, dtype: int64
In [289]:
data = data.drop(data[data["Patient experience national comparison"] == "Not Available"].index)
In [290]:
data["Patient experience national comparison"].value_counts(normalize=True)*100
Out[290]:
Below the national average      39.034046
Same as the national average    33.135392
Above the national average      27.830562
Name: Patient experience national comparison, dtype: float64
In [291]:
plt.figure(figsize=(15,7))
sns.countplot(x=data["Patient experience national comparison"], color="pink")
Out[291]:
<AxesSubplot:xlabel='Patient experience national comparison', ylabel='count'>

In [292]:
data["Effectiveness of care national comparison"].value_counts()
Out[292]:
Same as the national average    2231
Below the national average       166
Above the national average       124
Not Available                      5
Name: Effectiveness of care national comparison, dtype: int64
In [293]:
data = data.drop(data[data["Effectiveness of care national comparison"] == "Not Available"].index)
In [294]:
data["Effectiveness of care national comparison"].value_counts(normalize=True)*100
Out[294]:
Same as the national average    88.496628
Below the national average       6.584689
Above the national average       4.918683
Name: Effectiveness of care national comparison, dtype: float64
In [295]:
plt.figure(figsize=(15,7))
sns.countplot(x=data["Effectiveness of care national comparison"], palette="viridis")
Out[295]:
<AxesSubplot:xlabel='Effectiveness of care national comparison', ylabel='count'>

In [296]:
data["Timeliness of care national comparison"].value_counts()
Out[296]:
Same as the national average    1033
Below the national average       872
Above the national average       604
Not Available                     12
Name: Timeliness of care national comparison, dtype: int64
In [297]:
data = data.drop(data[data["Timeliness of care national comparison"] == "Not Available"].index)
In [298]:
data["Timeliness of care national comparison"].value_counts(normalize=True)*100
Out[298]:
Same as the national average    41.171782
Below the national average      34.754882
Above the national average      24.073336
Name: Timeliness of care national comparison, dtype: float64
In [299]:
plt.figure(figsize=(15,7))
sns.countplot(x=data["Timeliness of care national comparison"], palette="plasma")
Out[299]:
<AxesSubplot:xlabel='Timeliness of care national comparison', ylabel='count'>

In [300]:
data["Efficient use of medical imaging national comparison"].value_counts()
Out[300]:
Same as the national average    1635
Above the national average       356
Below the national average       306
Not Available                    212
Name: Efficient use of medical imaging national comparison, dtype: int64
In [301]:
data = data.drop(data[data["Efficient use of medical imaging national comparison"] == "Not Available"].index)
In [302]:
data["Efficient use of medical imaging national comparison"].value_counts(normalize=True)*100
Out[302]:
Same as the national average    71.179800
Above the national average      15.498476
Below the national average      13.321724
Name: Efficient use of medical imaging national comparison, dtype: float64
In [303]:
plt.figure(figsize=(15,7))
sns.countplot(x=data["Efficient use of medical imaging national comparison"], palette="cividis")
Out[303]:
<AxesSubplot:xlabel='Efficient use of medical imaging national comparison', ylabel='count'>

Explorative data analysis on CMS measures are complete. Need to drilldown in the basics before moving on to machine learning.
In [304]:
data["City"].value_counts()
Out[304]:
CHICAGO         18
HOUSTON         13
PHILADELPHIA    12
WASHINGTON      10
COLUMBUS        10
                ..
LENOIR           1
WAUKESHA         1
EDGEWOOD         1
NAPA             1
KERRVILLE        1
Name: City, Length: 1474, dtype: int64
The City data length has ~1400 and it is not feasible to create dummies to keep this feature. Putting them into fewer categories is not necessary as we have State and ZIP code.
In [305]:
data = data.drop("City", axis=1)
In [306]:
data["State"].value_counts(normalize=True)*100
Out[306]:
CA    7.575098
TX    7.139747
FL    6.181976
PA    4.832390
NY    4.788855
IL    4.658250
OH    4.614715
MI    3.569874
NC    3.134523
GA    3.090988
IN    2.873313
VA    2.742708
TN    2.655638
MO    2.568568
NJ    2.525033
WI    2.481498
KY    2.263822
MA    2.176752
AL    2.133217
WA    1.915542
OK    1.828472
LA    1.828472
SC    1.784937
AZ    1.697867
CO    1.610797
MN    1.567262
MS    1.436657
KS    1.262516
CT    1.175446
AR    1.175446
OR    1.131911
IA    1.088376
WV    1.001306
NV    0.740096
UT    0.696561
ME    0.696561
NE    0.696561
NM    0.609491
NH    0.522421
MT    0.435350
SD    0.435350
ID    0.435350
RI    0.391815
HI    0.348280
DC    0.304745
ND    0.261210
WY    0.261210
VT    0.261210
DE    0.261210
AK    0.130605
Name: State, dtype: float64
Looks like the top 3 states that have highest number of hospitals are CA, TX, and FL. Possibly due to the number of population, but I do not have the data to support this. It could be higher needs from sicker population in general, or just having excessive resources. These are just the preliminary impression.
In [307]:
plt.figure(figsize=(15,30))
state_order = data["State"].value_counts().sort_values(ascending=False).index
sns.countplot(y=data["State"], palette="coolwarm", order=state_order)
plt.title("Number of Hospitals in Each State")
plt.xlabel("Number of Hospitals")
Out[307]:
Text(0.5, 0, 'Number of Hospitals')

Let's focus on the top 10 states that have the highest number of hospitals for their overall ratings. At a glance, TX has the highest number of hospitals with 5-star rating compared to the rest. In contrast, NY has the highest number of hospitals with 1-star rating compared to her counterparts.
In [308]:
plt.figure(figsize=(12,18))
plt.title("Top 10 States")
state_order = data["State"].value_counts().sort_values(ascending=False)[:10].index
sns.countplot(y=data["State"], palette="coolwarm", order=state_order, data=data, hue="Hospital overall rating")
plt.legend(bbox_to_anchor=(1.2, 0.5), title="CMS Rating")
Out[308]:
<matplotlib.legend.Legend at 0x1e35591caf0>

Just out of curiousity, let's drilldown into the top 3 states.
In [372]:
len(data[data["State"] == "TX"])
Out[372]:
164
In [368]:
data[data["State"] == "TX"]["Hospital overall rating"].value_counts()
Out[368]:
3    78
4    46
2    30
5     8
1     2
Name: Hospital overall rating, dtype: int64
In [371]:
len(data[data["State"] == "NY"])
Out[371]:
110
In [369]:
data[data["State"] == "NY"]["Hospital overall rating"].value_counts()
Out[369]:
2    41
3    35
1    25
4     9
Name: Hospital overall rating, dtype: int64
In [373]:
len(data[data["State"] == "CA"])
Out[373]:
174
In [374]:
data[data["State"] == "CA"]["Hospital overall rating"].value_counts()
Out[374]:
3    70
2    56
4    36
1     8
5     4
Name: Hospital overall rating, dtype: int64
In [389]:
len(data[data["Hospital overall rating"] == 5])
Out[389]:
59
In [390]:
len(data[data["Hospital overall rating"] == 1])
Out[390]:
94
There are a total of 59 hospitals with 5-star rating out of total 2297 hospitals in the USA (after data cleansing).
Fun fact: Given the dataset, the probability of picking a 5-star rating hospital in the USA is 2.57%.
Fun fact: Given the dataset, the probability of picking a 1-star rating hospital in the USA is 4.1%.
In [388]:
high_star_prob = len(data[data["Hospital overall rating"] == 5])/len(data)*100
high_star_prob
Out[388]:
2.5685676969960816
In [391]:
low_star_prob = len(data[data["Hospital overall rating"] == 1])/len(data)*100
low_star_prob
Out[391]:
4.092294296909012
Fun fact: Given the dataset, the probability of picking a 5-star rating hospital in the USA that is from Texas is 0.35%.
In [413]:
texas_high_prob = len(data[data["State"] == "TX"][data["Hospital overall rating"] == 5])
texas_high_total = len(data[data["State"] == "TX"])
total_usa = len(data)
<ipython-input-413-f56fb7b6bfdc>:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
  texas_high_prob = len(data[data["State"] == "TX"][data["Hospital overall rating"] == 5])
In [414]:
texas_prob = (texas_high_prob/texas_high_total)*(texas_high_total/total_usa)*100
texas_prob
Out[414]:
0.34828036569438403
Fun fact: Given the dataset, the probability of picking a 1-star rating hospital in the USA that is from New York is 1.1%.
In [418]:
newyork_low_prob = len(data[data["State"] == "NY"][data["Hospital overall rating"] == 1])
newyork_low_total = len(data[data["State"] == "NY"])
total_usa = len(data)
<ipython-input-418-d6411bcf147b>:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
  newyork_low_prob = len(data[data["State"] == "NY"][data["Hospital overall rating"] == 1])
In [419]:
newyork_prob = (newyork_low_prob/newyork_low_total)*(newyork_low_total/total_usa)*100
newyork_prob
Out[419]:
1.0883761427949499
It's time to look into the data type and prepare for machine learning. Noted that all are objects except for ZIP code.
In [86]:
data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2297 entries, 0 to 4784
Data columns (total 13 columns):
 #   Column                                                Non-Null Count  Dtype 
---  ------                                                --------------  ----- 
 0   State                                                 2297 non-null   object
 1   ZIP Code                                              2297 non-null   int64 
 2   Hospital Type                                         2297 non-null   object
 3   Hospital Ownership                                    2297 non-null   object
 4   Emergency Services                                    2297 non-null   object
 5   Hospital overall rating                               2297 non-null   object
 6   Mortality national comparison                         2297 non-null   object
 7   Safety of care national comparison                    2297 non-null   object
 8   Readmission national comparison                       2297 non-null   object
 9   Patient experience national comparison                2297 non-null   object
 10  Effectiveness of care national comparison             2297 non-null   object
 11  Timeliness of care national comparison                2297 non-null   object
 12  Efficient use of medical imaging national comparison  2297 non-null   object
dtypes: int64(1), object(12)
memory usage: 331.2+ KB
Getting dummy data on features
In [87]:
clean_state = pd.get_dummies(data["State"], prefix="State_", drop_first=True, dtype=int)
data = pd.concat([data.drop("State", axis=1), clean_state], axis=1)
In [88]:
data.head(3)
Out[88]:
ZIP Code	Hospital Type	Hospital Ownership	Emergency Services	Hospital overall rating	Mortality national comparison	Safety of care national comparison	Readmission national comparison	Patient experience national comparison	Effectiveness of care national comparison	...	State__SD	State__TN	State__TX	State__UT	State__VA	State__VT	State__WA	State__WI	State__WV	State__WY
0	36301	Acute Care Hospitals	Government - Hospital District or Authority	Yes	3	Same as the national average	Above the national average	Same as the national average	Below the national average	Same as the national average	...	0	0	0	0	0	0	0	0	0	0
1	35957	Acute Care Hospitals	Government - Hospital District or Authority	Yes	3	Below the national average	Same as the national average	Above the national average	Same as the national average	Same as the national average	...	0	0	0	0	0	0	0	0	0	0
2	35631	Acute Care Hospitals	Government - Hospital District or Authority	Yes	2	Below the national average	Same as the national average	Same as the national average	Below the national average	Same as the national average	...	0	0	0	0	0	0	0	0	0	0
3 rows × 61 columns

In [89]:
data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2297 entries, 0 to 4784
Data columns (total 61 columns):
 #   Column                                                Non-Null Count  Dtype 
---  ------                                                --------------  ----- 
 0   ZIP Code                                              2297 non-null   int64 
 1   Hospital Type                                         2297 non-null   object
 2   Hospital Ownership                                    2297 non-null   object
 3   Emergency Services                                    2297 non-null   object
 4   Hospital overall rating                               2297 non-null   object
 5   Mortality national comparison                         2297 non-null   object
 6   Safety of care national comparison                    2297 non-null   object
 7   Readmission national comparison                       2297 non-null   object
 8   Patient experience national comparison                2297 non-null   object
 9   Effectiveness of care national comparison             2297 non-null   object
 10  Timeliness of care national comparison                2297 non-null   object
 11  Efficient use of medical imaging national comparison  2297 non-null   object
 12  State__AL                                             2297 non-null   int32 
 13  State__AR                                             2297 non-null   int32 
 14  State__AZ                                             2297 non-null   int32 
 15  State__CA                                             2297 non-null   int32 
 16  State__CO                                             2297 non-null   int32 
 17  State__CT                                             2297 non-null   int32 
 18  State__DC                                             2297 non-null   int32 
 19  State__DE                                             2297 non-null   int32 
 20  State__FL                                             2297 non-null   int32 
 21  State__GA                                             2297 non-null   int32 
 22  State__HI                                             2297 non-null   int32 
 23  State__IA                                             2297 non-null   int32 
 24  State__ID                                             2297 non-null   int32 
 25  State__IL                                             2297 non-null   int32 
 26  State__IN                                             2297 non-null   int32 
 27  State__KS                                             2297 non-null   int32 
 28  State__KY                                             2297 non-null   int32 
 29  State__LA                                             2297 non-null   int32 
 30  State__MA                                             2297 non-null   int32 
 31  State__ME                                             2297 non-null   int32 
 32  State__MI                                             2297 non-null   int32 
 33  State__MN                                             2297 non-null   int32 
 34  State__MO                                             2297 non-null   int32 
 35  State__MS                                             2297 non-null   int32 
 36  State__MT                                             2297 non-null   int32 
 37  State__NC                                             2297 non-null   int32 
 38  State__ND                                             2297 non-null   int32 
 39  State__NE                                             2297 non-null   int32 
 40  State__NH                                             2297 non-null   int32 
 41  State__NJ                                             2297 non-null   int32 
 42  State__NM                                             2297 non-null   int32 
 43  State__NV                                             2297 non-null   int32 
 44  State__NY                                             2297 non-null   int32 
 45  State__OH                                             2297 non-null   int32 
 46  State__OK                                             2297 non-null   int32 
 47  State__OR                                             2297 non-null   int32 
 48  State__PA                                             2297 non-null   int32 
 49  State__RI                                             2297 non-null   int32 
 50  State__SC                                             2297 non-null   int32 
 51  State__SD                                             2297 non-null   int32 
 52  State__TN                                             2297 non-null   int32 
 53  State__TX                                             2297 non-null   int32 
 54  State__UT                                             2297 non-null   int32 
 55  State__VA                                             2297 non-null   int32 
 56  State__VT                                             2297 non-null   int32 
 57  State__WA                                             2297 non-null   int32 
 58  State__WI                                             2297 non-null   int32 
 59  State__WV                                             2297 non-null   int32 
 60  State__WY                                             2297 non-null   int32 
dtypes: int32(49), int64(1), object(11)
memory usage: 752.9+ KB
In [90]:
dummy_column = data.iloc[:, 1:4]
In [91]:
clean_column = pd.get_dummies(dummy_column, drop_first=True, dtype=int)
data = pd.concat([data.drop(dummy_column, axis=1), clean_column], axis=1)
In [92]:
data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2297 entries, 0 to 4784
Data columns (total 68 columns):
 #   Column                                                          Non-Null Count  Dtype 
---  ------                                                          --------------  ----- 
 0   ZIP Code                                                        2297 non-null   int64 
 1   Hospital overall rating                                         2297 non-null   object
 2   Mortality national comparison                                   2297 non-null   object
 3   Safety of care national comparison                              2297 non-null   object
 4   Readmission national comparison                                 2297 non-null   object
 5   Patient experience national comparison                          2297 non-null   object
 6   Effectiveness of care national comparison                       2297 non-null   object
 7   Timeliness of care national comparison                          2297 non-null   object
 8   Efficient use of medical imaging national comparison            2297 non-null   object
 9   State__AL                                                       2297 non-null   int32 
 10  State__AR                                                       2297 non-null   int32 
 11  State__AZ                                                       2297 non-null   int32 
 12  State__CA                                                       2297 non-null   int32 
 13  State__CO                                                       2297 non-null   int32 
 14  State__CT                                                       2297 non-null   int32 
 15  State__DC                                                       2297 non-null   int32 
 16  State__DE                                                       2297 non-null   int32 
 17  State__FL                                                       2297 non-null   int32 
 18  State__GA                                                       2297 non-null   int32 
 19  State__HI                                                       2297 non-null   int32 
 20  State__IA                                                       2297 non-null   int32 
 21  State__ID                                                       2297 non-null   int32 
 22  State__IL                                                       2297 non-null   int32 
 23  State__IN                                                       2297 non-null   int32 
 24  State__KS                                                       2297 non-null   int32 
 25  State__KY                                                       2297 non-null   int32 
 26  State__LA                                                       2297 non-null   int32 
 27  State__MA                                                       2297 non-null   int32 
 28  State__ME                                                       2297 non-null   int32 
 29  State__MI                                                       2297 non-null   int32 
 30  State__MN                                                       2297 non-null   int32 
 31  State__MO                                                       2297 non-null   int32 
 32  State__MS                                                       2297 non-null   int32 
 33  State__MT                                                       2297 non-null   int32 
 34  State__NC                                                       2297 non-null   int32 
 35  State__ND                                                       2297 non-null   int32 
 36  State__NE                                                       2297 non-null   int32 
 37  State__NH                                                       2297 non-null   int32 
 38  State__NJ                                                       2297 non-null   int32 
 39  State__NM                                                       2297 non-null   int32 
 40  State__NV                                                       2297 non-null   int32 
 41  State__NY                                                       2297 non-null   int32 
 42  State__OH                                                       2297 non-null   int32 
 43  State__OK                                                       2297 non-null   int32 
 44  State__OR                                                       2297 non-null   int32 
 45  State__PA                                                       2297 non-null   int32 
 46  State__RI                                                       2297 non-null   int32 
 47  State__SC                                                       2297 non-null   int32 
 48  State__SD                                                       2297 non-null   int32 
 49  State__TN                                                       2297 non-null   int32 
 50  State__TX                                                       2297 non-null   int32 
 51  State__UT                                                       2297 non-null   int32 
 52  State__VA                                                       2297 non-null   int32 
 53  State__VT                                                       2297 non-null   int32 
 54  State__WA                                                       2297 non-null   int32 
 55  State__WI                                                       2297 non-null   int32 
 56  State__WV                                                       2297 non-null   int32 
 57  State__WY                                                       2297 non-null   int32 
 58  Hospital Type_Critical Access Hospitals                         2297 non-null   int32 
 59  Hospital Ownership_Government - Hospital District or Authority  2297 non-null   int32 
 60  Hospital Ownership_Government - Local                           2297 non-null   int32 
 61  Hospital Ownership_Government - State                           2297 non-null   int32 
 62  Hospital Ownership_Physician                                    2297 non-null   int32 
 63  Hospital Ownership_Proprietary                                  2297 non-null   int32 
 64  Hospital Ownership_Voluntary non-profit - Church                2297 non-null   int32 
 65  Hospital Ownership_Voluntary non-profit - Other                 2297 non-null   int32 
 66  Hospital Ownership_Voluntary non-profit - Private               2297 non-null   int32 
 67  Emergency Services_Yes                                          2297 non-null   int32 
dtypes: int32(59), int64(1), object(8)
memory usage: 788.8+ KB
In [93]:
dummy_columns = data.iloc[:, 2:9]
In [94]:
clean_columns = pd.get_dummies(dummy_columns, drop_first=True, dtype=int)
data = pd.concat([data.drop(dummy_columns, axis=1), clean_columns], axis=1)
In [95]:
data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2297 entries, 0 to 4784
Data columns (total 75 columns):
 #   Column                                                                             Non-Null Count  Dtype 
---  ------                                                                             --------------  ----- 
 0   ZIP Code                                                                           2297 non-null   int64 
 1   Hospital overall rating                                                            2297 non-null   object
 2   State__AL                                                                          2297 non-null   int32 
 3   State__AR                                                                          2297 non-null   int32 
 4   State__AZ                                                                          2297 non-null   int32 
 5   State__CA                                                                          2297 non-null   int32 
 6   State__CO                                                                          2297 non-null   int32 
 7   State__CT                                                                          2297 non-null   int32 
 8   State__DC                                                                          2297 non-null   int32 
 9   State__DE                                                                          2297 non-null   int32 
 10  State__FL                                                                          2297 non-null   int32 
 11  State__GA                                                                          2297 non-null   int32 
 12  State__HI                                                                          2297 non-null   int32 
 13  State__IA                                                                          2297 non-null   int32 
 14  State__ID                                                                          2297 non-null   int32 
 15  State__IL                                                                          2297 non-null   int32 
 16  State__IN                                                                          2297 non-null   int32 
 17  State__KS                                                                          2297 non-null   int32 
 18  State__KY                                                                          2297 non-null   int32 
 19  State__LA                                                                          2297 non-null   int32 
 20  State__MA                                                                          2297 non-null   int32 
 21  State__ME                                                                          2297 non-null   int32 
 22  State__MI                                                                          2297 non-null   int32 
 23  State__MN                                                                          2297 non-null   int32 
 24  State__MO                                                                          2297 non-null   int32 
 25  State__MS                                                                          2297 non-null   int32 
 26  State__MT                                                                          2297 non-null   int32 
 27  State__NC                                                                          2297 non-null   int32 
 28  State__ND                                                                          2297 non-null   int32 
 29  State__NE                                                                          2297 non-null   int32 
 30  State__NH                                                                          2297 non-null   int32 
 31  State__NJ                                                                          2297 non-null   int32 
 32  State__NM                                                                          2297 non-null   int32 
 33  State__NV                                                                          2297 non-null   int32 
 34  State__NY                                                                          2297 non-null   int32 
 35  State__OH                                                                          2297 non-null   int32 
 36  State__OK                                                                          2297 non-null   int32 
 37  State__OR                                                                          2297 non-null   int32 
 38  State__PA                                                                          2297 non-null   int32 
 39  State__RI                                                                          2297 non-null   int32 
 40  State__SC                                                                          2297 non-null   int32 
 41  State__SD                                                                          2297 non-null   int32 
 42  State__TN                                                                          2297 non-null   int32 
 43  State__TX                                                                          2297 non-null   int32 
 44  State__UT                                                                          2297 non-null   int32 
 45  State__VA                                                                          2297 non-null   int32 
 46  State__VT                                                                          2297 non-null   int32 
 47  State__WA                                                                          2297 non-null   int32 
 48  State__WI                                                                          2297 non-null   int32 
 49  State__WV                                                                          2297 non-null   int32 
 50  State__WY                                                                          2297 non-null   int32 
 51  Hospital Type_Critical Access Hospitals                                            2297 non-null   int32 
 52  Hospital Ownership_Government - Hospital District or Authority                     2297 non-null   int32 
 53  Hospital Ownership_Government - Local                                              2297 non-null   int32 
 54  Hospital Ownership_Government - State                                              2297 non-null   int32 
 55  Hospital Ownership_Physician                                                       2297 non-null   int32 
 56  Hospital Ownership_Proprietary                                                     2297 non-null   int32 
 57  Hospital Ownership_Voluntary non-profit - Church                                   2297 non-null   int32 
 58  Hospital Ownership_Voluntary non-profit - Other                                    2297 non-null   int32 
 59  Hospital Ownership_Voluntary non-profit - Private                                  2297 non-null   int32 
 60  Emergency Services_Yes                                                             2297 non-null   int32 
 61  Mortality national comparison_Below the national average                           2297 non-null   int32 
 62  Mortality national comparison_Same as the national average                         2297 non-null   int32 
 63  Safety of care national comparison_Below the national average                      2297 non-null   int32 
 64  Safety of care national comparison_Same as the national average                    2297 non-null   int32 
 65  Readmission national comparison_Below the national average                         2297 non-null   int32 
 66  Readmission national comparison_Same as the national average                       2297 non-null   int32 
 67  Patient experience national comparison_Below the national average                  2297 non-null   int32 
 68  Patient experience national comparison_Same as the national average                2297 non-null   int32 
 69  Effectiveness of care national comparison_Below the national average               2297 non-null   int32 
 70  Effectiveness of care national comparison_Same as the national average             2297 non-null   int32 
 71  Timeliness of care national comparison_Below the national average                  2297 non-null   int32 
 72  Timeliness of care national comparison_Same as the national average                2297 non-null   int32 
 73  Efficient use of medical imaging national comparison_Below the national average    2297 non-null   int32 
 74  Efficient use of medical imaging national comparison_Same as the national average  2297 non-null   int32 
dtypes: int32(73), int64(1), object(1)
memory usage: 788.8+ KB
Last, our y-variable will be converted from object to int. Once again, rechecking all columns.
In [96]:
data["Hospital overall rating"] = data["Hospital overall rating"].astype(str).astype(int)
In [97]:
data.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2297 entries, 0 to 4784
Data columns (total 75 columns):
 #   Column                                                                             Non-Null Count  Dtype
---  ------                                                                             --------------  -----
 0   ZIP Code                                                                           2297 non-null   int64
 1   Hospital overall rating                                                            2297 non-null   int32
 2   State__AL                                                                          2297 non-null   int32
 3   State__AR                                                                          2297 non-null   int32
 4   State__AZ                                                                          2297 non-null   int32
 5   State__CA                                                                          2297 non-null   int32
 6   State__CO                                                                          2297 non-null   int32
 7   State__CT                                                                          2297 non-null   int32
 8   State__DC                                                                          2297 non-null   int32
 9   State__DE                                                                          2297 non-null   int32
 10  State__FL                                                                          2297 non-null   int32
 11  State__GA                                                                          2297 non-null   int32
 12  State__HI                                                                          2297 non-null   int32
 13  State__IA                                                                          2297 non-null   int32
 14  State__ID                                                                          2297 non-null   int32
 15  State__IL                                                                          2297 non-null   int32
 16  State__IN                                                                          2297 non-null   int32
 17  State__KS                                                                          2297 non-null   int32
 18  State__KY                                                                          2297 non-null   int32
 19  State__LA                                                                          2297 non-null   int32
 20  State__MA                                                                          2297 non-null   int32
 21  State__ME                                                                          2297 non-null   int32
 22  State__MI                                                                          2297 non-null   int32
 23  State__MN                                                                          2297 non-null   int32
 24  State__MO                                                                          2297 non-null   int32
 25  State__MS                                                                          2297 non-null   int32
 26  State__MT                                                                          2297 non-null   int32
 27  State__NC                                                                          2297 non-null   int32
 28  State__ND                                                                          2297 non-null   int32
 29  State__NE                                                                          2297 non-null   int32
 30  State__NH                                                                          2297 non-null   int32
 31  State__NJ                                                                          2297 non-null   int32
 32  State__NM                                                                          2297 non-null   int32
 33  State__NV                                                                          2297 non-null   int32
 34  State__NY                                                                          2297 non-null   int32
 35  State__OH                                                                          2297 non-null   int32
 36  State__OK                                                                          2297 non-null   int32
 37  State__OR                                                                          2297 non-null   int32
 38  State__PA                                                                          2297 non-null   int32
 39  State__RI                                                                          2297 non-null   int32
 40  State__SC                                                                          2297 non-null   int32
 41  State__SD                                                                          2297 non-null   int32
 42  State__TN                                                                          2297 non-null   int32
 43  State__TX                                                                          2297 non-null   int32
 44  State__UT                                                                          2297 non-null   int32
 45  State__VA                                                                          2297 non-null   int32
 46  State__VT                                                                          2297 non-null   int32
 47  State__WA                                                                          2297 non-null   int32
 48  State__WI                                                                          2297 non-null   int32
 49  State__WV                                                                          2297 non-null   int32
 50  State__WY                                                                          2297 non-null   int32
 51  Hospital Type_Critical Access Hospitals                                            2297 non-null   int32
 52  Hospital Ownership_Government - Hospital District or Authority                     2297 non-null   int32
 53  Hospital Ownership_Government - Local                                              2297 non-null   int32
 54  Hospital Ownership_Government - State                                              2297 non-null   int32
 55  Hospital Ownership_Physician                                                       2297 non-null   int32
 56  Hospital Ownership_Proprietary                                                     2297 non-null   int32
 57  Hospital Ownership_Voluntary non-profit - Church                                   2297 non-null   int32
 58  Hospital Ownership_Voluntary non-profit - Other                                    2297 non-null   int32
 59  Hospital Ownership_Voluntary non-profit - Private                                  2297 non-null   int32
 60  Emergency Services_Yes                                                             2297 non-null   int32
 61  Mortality national comparison_Below the national average                           2297 non-null   int32
 62  Mortality national comparison_Same as the national average                         2297 non-null   int32
 63  Safety of care national comparison_Below the national average                      2297 non-null   int32
 64  Safety of care national comparison_Same as the national average                    2297 non-null   int32
 65  Readmission national comparison_Below the national average                         2297 non-null   int32
 66  Readmission national comparison_Same as the national average                       2297 non-null   int32
 67  Patient experience national comparison_Below the national average                  2297 non-null   int32
 68  Patient experience national comparison_Same as the national average                2297 non-null   int32
 69  Effectiveness of care national comparison_Below the national average               2297 non-null   int32
 70  Effectiveness of care national comparison_Same as the national average             2297 non-null   int32
 71  Timeliness of care national comparison_Below the national average                  2297 non-null   int32
 72  Timeliness of care national comparison_Same as the national average                2297 non-null   int32
 73  Efficient use of medical imaging national comparison_Below the national average    2297 non-null   int32
 74  Efficient use of medical imaging national comparison_Same as the national average  2297 non-null   int32
dtypes: int32(74), int64(1)
memory usage: 779.9 KB
Preprocessing data for machine learning models. Importing sci-kit learn libraries.
In [101]:
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
In [102]:
scaler = MinMaxScaler()
Creating variables for X and y. Our target is to predict hospital rating, thus, our y variable is "Hospital overal rating" column. The rest will be our X variable.
In [104]:
X = data.drop("Hospital overall rating", axis=1)
y = data["Hospital overall rating"]
Setting up the data for training and testing with 20% of the data as the test size. Random state will be used as one of the parameters so the random generation will produce the same sequence each time running the algorithm. An arbitrary number 42 will be used as the random state (I heard it is THE number for Life, universe, and everything :)
In [108]:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
In [109]:
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
K-nearest neighbor
K-nearest neighbor - a random number for the K-value or the n_neighbors of 10 will be used.
In [110]:
from sklearn.neighbors import KNeighborsClassifier
In [111]:
neighbor = KNeighborsClassifier(n_neighbors=10)
In [112]:
neighbor.fit(X_train, y_train)
Out[112]:
KNeighborsClassifier(n_neighbors=10)
In [113]:
knnpredict = neighbor.predict(X_test)
Training and prediction complete. Now let's evaluate.
In [114]:
from sklearn.metrics import classification_report, confusion_matrix
In [115]:
print(classification_report(y_test, knnpredict))
print(confusion_matrix(y_test, knnpredict))
              precision    recall  f1-score   support

           1       0.47      0.41      0.44        17
           2       0.62      0.57      0.59       105
           3       0.62      0.71      0.66       212
           4       0.56      0.52      0.54       113
           5       0.50      0.08      0.13        13

    accuracy                           0.60       460
   macro avg       0.55      0.46      0.47       460
weighted avg       0.60      0.60      0.59       460

[[  7   9   1   0   0]
 [  6  60  37   2   0]
 [  2  25 150  35   0]
 [  0   3  50  59   1]
 [  0   0   3   9   1]]
The accurate is 60% with n_neighbors set as 10, which was a random number. Noted that precision and recall values are almost at 50% with recall value for rating 5 is only at 8%, and F1-score of 13%. However, the test size for rating 5 is only 13, which must be considered.
The elbow method will be used to evaluate what number is the best K-value in range 1 to 40, which is also arbitrary.

In [116]:
error_rate = []

for i in range(1,40):
    neighbor = KNeighborsClassifier(n_neighbors = i)
    neighbor.fit(X_train, y_train)
    knnpredict = neighbor.predict(X_test)
    error_rate.append(np.mean(knnpredict != y_test))
The graph below shows the error rate trend in the elbow method. Noted that K-value 15 to 35 gives as a steady trend, hovering at ~0.38 error rate. It is logical to pick the K-value that gives us the lowest error rate. In this case, I will pick 8 since it is the starting point of downtrend.
In [117]:
plt.figure(figsize=(15,7))
plt.plot(range(1,40), error_rate, color="blue", ls="dashed", marker="o", markerfacecolor="red", markersize=10)
plt.title("Error Rate")
plt.xlabel("K-value")
plt.ylabel("Error Rate")
Out[117]:
Text(0, 0.5, 'Error Rate')

In [128]:
neighbor = KNeighborsClassifier(n_neighbors=8)
neighbor.fit(X_train, y_train)
knnpredict = neighbor.predict(X_test)
In [129]:
print(classification_report(y_test, knnpredict))
print(confusion_matrix(y_test, knnpredict))
              precision    recall  f1-score   support

           1       0.53      0.47      0.50        17
           2       0.64      0.64      0.64       105
           3       0.64      0.70      0.67       212
           4       0.60      0.54      0.57       113
           5       0.50      0.15      0.24        13

    accuracy                           0.62       460
   macro avg       0.58      0.50      0.52       460
weighted avg       0.62      0.62      0.62       460

[[  8   5   4   0   0]
 [  5  67  32   1   0]
 [  2  30 149  31   0]
 [  0   3  47  61   2]
 [  0   0   2   9   2]]
The KNN model using elbow method resulted in a slight increase in accuracy, from 60% to 62%. There is an improvement in recall of 15% in rating 5 and F1-score of 24%, which is much better than before IMO. Moving on to Support Vector Machines, which I think it is a close counterpart to KNN.
Support Vector Machines
In [130]:
from sklearn.svm import SVC
In [143]:
support = SVC(random_state=42)
In [144]:
support.fit(X_train, y_train)
Out[144]:
SVC(random_state=42)
In [145]:
support_predict = support.predict(X_test)
In [146]:
print(classification_report(y_test, support_predict))
print(confusion_matrix(y_test, support_predict))
              precision    recall  f1-score   support

           1       1.00      0.06      0.11        17
           2       0.68      0.67      0.67       105
           3       0.71      0.80      0.75       212
           4       0.67      0.71      0.69       113
           5       0.00      0.00      0.00        13

    accuracy                           0.70       460
   macro avg       0.61      0.45      0.45       460
weighted avg       0.69      0.70      0.67       460

[[  1  16   0   0   0]
 [  0  70  35   0   0]
 [  0  17 169  26   0]
 [  0   0  33  80   0]
 [  0   0   0  13   0]]
C:\Users\Adrian\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1221: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
The SVM model yields 70% accuracy. Noted that it has 100% precision on rating 1 but it failed to recognize anything on rating 5.
Random Forest
In [164]:
from sklearn.ensemble import RandomForestClassifier
In [210]:
random_forest = RandomForestClassifier(n_estimators=260, bootstrap=True, random_state=42)
random_forest.fit(X_train, y_train)
Out[210]:
RandomForestClassifier(n_estimators=260, random_state=42)
In [211]:
random_forest_predict = random_forest.predict(X_test)
In [212]:
print(classification_report(y_test, random_forest_predict))
print(confusion_matrix(y_test, random_forest_predict))
              precision    recall  f1-score   support

           1       0.40      0.24      0.30        17
           2       0.67      0.58      0.62       105
           3       0.67      0.80      0.73       212
           4       0.64      0.58      0.61       113
           5       0.25      0.08      0.12        13

    accuracy                           0.66       460
   macro avg       0.53      0.46      0.48       460
weighted avg       0.64      0.66      0.64       460

[[  4  13   0   0   0]
 [  5  61  39   0   0]
 [  1  16 170  25   0]
 [  0   1  43  66   3]
 [  0   0   0  12   1]]
The Random Forest model yields a 66% accuracy with at least some correct predictions on rating 5 compared to SVM.
In [236]:
report = [["Support Vector Machines", 0.70, 0.69, 0.70, 0.67], ["Random Forest", 0.66, 0.64, 0.66, 0.64], 
          ["K-nearest neighbor", 0.62, 0.62, 0.62, 0.62]]
overall_result = pd.DataFrame(report, columns=["Model", "Accuracy Score", "Precision", "Recall", "F1-score"])
overall_result.sort_values("F1-score", ascending=False)
Out[236]:
Model	Accuracy Score	Precision	Recall	F1-score
0	Support Vector Machines	0.70	0.69	0.70	0.67
1	Random Forest	0.66	0.64	0.66	0.64
2	K-nearest neighbor	0.62	0.62	0.62	0.62
Of the 3 machine learning models, SVM yields the highest accuracy and F1-score, followed by Random Forest, and K-nearest neighbor. The SVM model also scored 1.0 for rating 1. However, SVM failed to recognize any rating of 5. I believe the model can be better if we have more balanced data (recalling the majority of the y-variable was rating of 3 and extreme low count on rating 1 and 5).
