# Business Problem : make recommendations to the customer at the basket stage.

# Dataset : Online Retail II


# Libraries

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Statistics #####################")
    print(dataframe.describe().T)
    print("##################### Null #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def grab_col_names(df, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

    num_but_cat = [col for col in df.columns if df[col].nunique()<cat_th and df[col].dtypes != "O"]

    cat_but_car = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique()>cat_th]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]

    num_cols =[col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(df):
    df.dropna(inplace = True)
    df = df[~df["Invoice"].astype(str).str.contains("C", na=False)]
    df = df[df['Price'] > 0]
    df = df[df['Quantity'] > 0]
    df = df[df['StockCode'] != 'POST']
    replace_with_thresholds(df, "Quantity")
    replace_with_thresholds(df, "Price")
    return df

def arl_data_prep (df, id = False ):
    if id:
        return df.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return df.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code=[]):
    product_name = dataframe[dataframe["StockCode"]== stock_code][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(df,id=True, country=''):
    if country !='':
          df = df[df['Country'] == country]
    else :
          df = df
    df = arl_data_prep(df, id)
    frequent_itemsets = apriori(df, min_support= 0.01, use_colnames=True)
    frequent_itemsets.sort_values("support", ascending=False).head(5)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

def print_rec (rec_list):
    for i in rec_list:
        print (check_id(df, i))

if __name__ == '__main__':
    # Reading the dataset
    # pip install xlrd
    df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
    df = df_.copy()
    df.head()

    # EDA
    check_df(df)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # Preparing data
    df = retail_data_prep(df)

    # Extracting Association Rules
    rules = create_rules(df, country='Germany')

    #Making recommendations
    arl_recommender(rules, 21987, 2)


