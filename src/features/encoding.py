from sklearn.preprocessing import LabelEncoder


def encode_series(values):
    encoder = LabelEncoder()
    return encoder.fit_transform(values.astype(str)), encoder
