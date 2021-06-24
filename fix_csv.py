import pandas as pd

df = pd.read_csv('/media/zac/12TB Drive/covid-detector/meta/validation.csv', header=None,
                 names=['img_name', 'x_min', 'y_min', 'x_max', 'y_max', 'opacity'])
                 #dtype={'img_name':str, 'x_min':int, 'y_min':int, 'x_max':int, 'y_max':int, 'opacity':str})
# /media/zac/12TB Drive/covid-detector/extracted_images/img_subset/
# /media/zac/12TB Drive/covid-detector/extracted_images/jpgs/01d83be76fbb.jpg
df = df.applymap(lambda x: str(x).replace("jpgs", "img_subset"))
# nan_value = float("NaN")
# df.replace("", nan_value, inplace=True)
# df.dropna(inplace=True)
#df = df.dropna()
print(df['x_min'])
df['x_min'] = pd.to_numeric(df['x_min'], errors='coerce')
print(df['x_min'])
df['y_min'] = pd.to_numeric(df['y_min']).astype(int)
df['x_max'] = pd.to_numeric(df['x_max']).astype(int)
df['y_max'] = pd.to_numeric(df['y_max']).astype(int)

df.to_csv('/media/zac/12TB Drive/covid-detector/meta/validation_converted.csv', header=None, index=None)