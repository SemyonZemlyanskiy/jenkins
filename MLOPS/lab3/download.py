import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def download_data():
    df = pd.read_csv('vehicles_dataset.csv')
    return df

def clear_data(df):

    numerical_cols = ['price', 'year', 'odometer']

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] > lower) & (df[col] < upper)]

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.columns:
        
        vc = df[col].value_counts()
        if len(vc) > 0:
            if vc.values[0] / df.shape[0] > 0.9:
                df = df.drop(col, axis = 1)

    cat_columns = df.select_dtypes(include='object').columns

    for col in cat_columns:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < 0.05].index
        df[col] = df[col].apply(lambda x: 'Other' if x in rare or x == 'other' else x)


    rename_dict = {}
    for col in df.columns:
        rename_dict[col] = col.replace(' ', '_')
    df.rename(columns=rename_dict, inplace=True)

    df.drop(['id', 'VIN', 'state', 'lat', 'long', 'price_category', 'url', 'region', 'region_url', 'model', 'image_url','description', 'county', 'posting_date'], axis = 1, inplace=True)

    nominal_feats = ['manufacturer', 'fuel', 'type', 'paint_color']

    ohe = OneHotEncoder(sparse_output=False)

    encoded = ohe.fit_transform(df[nominal_feats])

    df[ohe.get_feature_names_out()] = encoded

    df = df.drop(columns=nominal_feats)

    not_nominal_features = ['condition', 'cylinders', 'transmission', 'drive', 'size']

    features_map = {
        'condition': {
            'Other': 0,
            'Good': 1,
            'excellent': 2
        },
        'cylinders': {
            'Other': 0,
            '4 cylinders': 1,
            '6 cylinders': 2,
            '8 cylinders': 3
        },
        'transmission': {
            'Other': 0,
            'automatic': 1
        },
        'drive': {
            'fwd': 0,
            '4wd': 1,
            'rwd': 1.2
        },
        'size': {
            'full-size': 2,
            'mid-size': 1,
            'Other': 0
        }
    }

    for col in not_nominal_features:
        df[f'{col}_encoded'] = df[col].map(features_map[col])
        df = df.drop(col, axis = 1)

    df = df.dropna()

    df.to_csv('df_clear.csv')

df = download_data()
clear_data(df)