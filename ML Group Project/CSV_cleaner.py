import csv
import pandas as pd



#Read in URLs and CSV files.
df = pd.read_csv("facebook-fact-check.csv")
print(df.head())

#Must have more than 5 comments
#Filter and output
for i in range(len(df.index)):
    if(df.iloc[i,11] >= 5 ):
        print(df.iloc[i,1] , " and " , df.iloc[i,4])
        df.to_csv("Clean_Posts")

    


