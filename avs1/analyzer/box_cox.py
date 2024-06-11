# import pandas as pd
# from scipy import stats
# from sklearn.preprocessing import StandardScaler
#
# # Чтение данных из файла data.csv
# rfm_data = pd.read_csv(r".\data\data.csv")
#
# # Преобразование Box-Cox к столбцу size_kb
# rfm_data['size_kb'], _ = stats.boxcox(rfm_data['size_kb'])
#
#
# # Нормализация данных с использованием StandardScaler
# scaler = StandardScaler()
# rfm_norm = scaler.fit_transform(rfm_data.drop(columns=['user_id', 'mac_address', 'user_dp_id']))
# rfm_norm = pd.DataFrame(rfm_norm, index=rfm_data.index, columns=rfm_data.columns.drop(['user_id', 'mac_address', 'user_dp_id']))
#
# print(rfm_norm)

# import pandas as pd
# from scipy import stats
# from sklearn.preprocessing import StandardScaler
# import uuid
#
# # Загрузка данных из файла data.csv
# data = pd.read_csv(r".\data\data.csv")
#
# def boxcox_df(x):
#     x_boxcox, _ = stats.boxcox(x)
#     return x_boxcox
#
# # Применение boxcox_df ко всем столбцам, кроме "user_id" и "mac_address"
# columns_to_transform = [col for col in data.columns if col not in ["user_id", "mac_address", "@timestamp"]]
# data_boxcox = data[columns_to_transform].apply(boxcox_df, axis=0)
#
# # Нормализация данных используя StandardScaler
# scaler = StandardScaler()
# scaler.fit(data_boxcox)
# data_norm = scaler.transform(data_boxcox)
# data_norm = pd.DataFrame(data_norm, index=data_boxcox.index, columns=data_boxcox.columns)
#
# # Добавление новых значений в столбец "mac_address"
# data["mac_address"] = [str(uuid.uuid4()) for _ in range(len(data))]
#
# # Вывод обновленных данных
# print(data)


# import pandas as pd
# from scipy import stats
# from sklearn.preprocessing import StandardScaler
#
# # Чтение данных из файла data.csv
# rfm_data = pd.read_csv(r".\data\data.csv")
#
# # Преобразование Box-Cox к столбцу size_kb
# rfm_data['size_kb'], _ = stats.boxcox(rfm_data['size_kb'])
#
# # Преобразование Box-Cox к столбцу @timestamp
# rfm_data['@timestamp'], _ = stats.boxcox(pd.to_datetime(rfm_data['@timestamp']).astype('int64'))
#
# # Нормализация данных с использованием StandardScaler
# scaler = StandardScaler()
# columns_to_normalize = ['size_kb', '@timestamp']
# rfm_data[columns_to_normalize] = scaler.fit_transform(rfm_data[columns_to_normalize])
# rfm_data.to_csv(r".\data\boxcoxed_data.csv", index=False)
#
# print(rfm_data)

# import pandas as pd
# from scipy.stats import boxcox
# from sklearn.preprocessing import StandardScaler
#
#
# def preprocess_network_data(input_file, output_file):
#     """
#     Преобразует данные из CSV-файла input_file с использованием Box-Cox преобразования и нормализации,
#     затем сохраняет результат в CSV-файл output_file.
#
#     Параметры:
#         input_file (str): Путь к входному CSV-файлу с данными.
#         output_file (str): Путь к выходному CSV-файлу, куда будут сохранены преобразованные данные.
#     """
#     # Чтение данных
#     network_data = pd.read_csv(input_file)
#
#     # Преобразование строковых значений в числовые для столбцов system.network.out.bytes и system.network.in.bytes
#     network_data['system.network.out.bytes'] = pd.to_numeric(network_data['system.network.out.bytes'], errors='coerce')
#     network_data['system.network.in.bytes'] = pd.to_numeric(network_data['system.network.in.bytes'], errors='coerce')
#
#     # Преобразование Box-Cox к столбцам system.network.out.bytes и system.network.in.bytes
#     network_data['system.network.out.bytes'] = boxcox(network_data['system.network.out.bytes'] + 1)[
#         0]  # Добавляем 1, чтобы избежать нулей
#     network_data['system.network.in.bytes'] = boxcox(network_data['system.network.in.bytes'] + 1)[0]
#
#     # Преобразование Box-Cox к столбцу @timestamp
#     network_data['@timestamp'] = boxcox(pd.to_datetime(network_data['@timestamp']).astype('int64') + 1)[0]
#
#     # Нормализация данных с использованием StandardScaler
#     scaler = StandardScaler()
#     columns_to_normalize = ['system.network.out.bytes', 'system.network.in.bytes', '@timestamp']
#     network_data[columns_to_normalize] = scaler.fit_transform(network_data[columns_to_normalize])
#
#     # Сохранение преобразованных данных
#     network_data.to_csv(output_file, index=False)
#
#     return network_data
#
#
# # Пример использования функции
# preprocessed_network_data = preprocess_network_data(r".\data\data2.csv", r".\data\boxcoxed_data2.csv")
# print(preprocessed_network_data)

# _______________________________________________________________________________________________________________________

import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_file, output_file):

    # Чтение данных
    rfm_data = pd.read_csv(input_file)

    # Преобразование Box-Cox к столбцу size_kb
    rfm_data['size_kb'], _ = stats.boxcox(rfm_data['size_kb'])

    # Преобразование Box-Cox к столбцу @timestamp
    rfm_data['@timestamp'], _ = stats.boxcox(pd.to_datetime(rfm_data['@timestamp']).astype('int64'))

    # Нормализация данных с использованием StandardScaler
    scaler = StandardScaler()
    columns_to_normalize = ['size_kb', '@timestamp']
    rfm_data[columns_to_normalize] = scaler.fit_transform(rfm_data[columns_to_normalize])

    # Сохранение преобразованных данных
    rfm_data.to_csv(output_file, index=False)

    return rfm_data


# # Пример использования функции
# preprocessed_data = preprocess_data(r".\data\data2.csv", r".\data\boxcoxed_data2.csv")
# print(preprocessed_data)





