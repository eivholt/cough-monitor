import pandas as pd
import time
import numpy as np

start = time.process_time()

df = pd.read_csv("..\\dataset\\kaggle covid-19 cough audio classification\\archive\\metadata_compiled.csv")
print("Load csv (s): ", round(time.process_time() - start, 2))

pd.set_option('display.max_columns', None)
#print(df.head(5))

print("Items: ", len(df))

def printValues(df):
    print("status", df.status.unique())
    print("quality_1", df.quality_1.unique())
    print("quality_2", df.quality_2.unique())
    print("quality_3", df.quality_3.unique())
    print("quality_4", df.quality_4.unique())
    print("cough_type_1", df.cough_type_1.unique())
    print("cough_type_2", df.cough_type_2.unique())
    print("cough_type_3", df.cough_type_3.unique())
    print("cough_type_4", df.cough_type_4.unique())
    print("dyspnea_1", df.dyspnea_1.unique())
    print("dyspnea_2", df.dyspnea_2.unique())
    print("dyspnea_3", df.dyspnea_3.unique())
    print("dyspnea_4", df.dyspnea_4.unique())
    print("cough_type_1", df.cough_type_1.unique())
    print("cough_type_2", df.cough_type_2.unique())
    print("cough_type_3", df.cough_type_3.unique())
    print("cough_type_4", df.cough_type_4.unique())
    print("choking_1", df.choking_1.unique())
    print("choking_2", df.choking_2.unique())
    print("choking_3", df.choking_3.unique())
    print("choking_4", df.choking_4.unique())
    print("congestion_1", df.congestion_1.unique())
    print("congestion_2", df.congestion_2.unique())
    print("congestion_3", df.congestion_3.unique())
    print("congestion_4", df.congestion_4.unique())
    print("cough_type_1", df.cough_type_1.unique())
    print("cough_type_2", df.cough_type_2.unique())
    print("cough_type_3", df.cough_type_3.unique())
    print("cough_type_4", df.cough_type_4.unique())
    print("diagnosis_1", df.diagnosis_1.unique())
    print("diagnosis_2", df.diagnosis_2.unique())
    print("diagnosis_3", df.diagnosis_3.unique())
    print("diagnosis_4", df.diagnosis_4.unique())
    print("nothing_1", df.nothing_1.unique())
    print("nothing_2", df.nothing_2.unique())
    print("nothing_3", df.nothing_3.unique())
    print("nothing_4", df.nothing_4.unique())
    print("quality_1", df.quality_1.unique())
    print("quality_2", df.quality_2.unique())
    print("quality_3", df.quality_3.unique())
    print("quality_4", df.quality_4.unique())
    print("respiratory_condition", df.respiratory_condition.unique())
    print("severity_1", df.severity_1.unique())
    print("severity_2", df.severity_2.unique())
    print("severity_3", df.severity_3.unique())
    print("severity_4", df.severity_4.unique())
    print("stridor_1", df.stridor_1.unique())
    print("stridor_2", df.stridor_2.unique())
    print("stridor_3", df.stridor_3.unique())
    print("stridor_4", df.stridor_4.unique())
    print("wheezing_1", df.wheezing_1.unique())
    print("wheezing_2", df.wheezing_2.unique())
    print("wheezing_3", df.wheezing_3.unique())
    print("wheezing_4", df.wheezing_4.unique())

#printValues(df)
df = df[df["cough_detected"] >= 0.99]
print("\ncough_detected >= 100%: ", len(df))

status_options = ["healthy"]
df = df[df["status"].isin(status_options)]
print("\nseverity: ", status_options, len(df))

respiratory_condition_options = [False]
df = df[df["respiratory_condition"].isin(respiratory_condition_options)]
print("\nrespiratory_condition: ", respiratory_condition_options, len(df))

severity_options = ["mild", "pseudocough", np.nan]
df = df[df["severity_1"].isin(severity_options) & df["severity_2"].isin(severity_options) & df["severity_3"].isin(severity_options) & df["severity_4"].isin(severity_options)]
print("\nseverity: ", severity_options, len(df))

wheezing_options = [False, np.nan]
df = df[df["wheezing_1"].isin(wheezing_options) & df["wheezing_2"].isin(wheezing_options) & df["wheezing_3"].isin(wheezing_options) & df["wheezing_4"].isin(wheezing_options)]
print("\nwheezing: ", wheezing_options, len(df))

diagnosis_options = ["healthy_cough", np.nan]
df = df[df["diagnosis_1"].isin(diagnosis_options) & df["diagnosis_2"].isin(diagnosis_options) & df["diagnosis_3"].isin(diagnosis_options) & df["diagnosis_4"].isin(diagnosis_options)]
print("\ndiagnosis: ", diagnosis_options, len(df))

cough_type_options = ["dry", np.nan]
df = df[df["cough_type_1"].isin(cough_type_options) & df["cough_type_2"].isin(cough_type_options) & df["cough_type_3"].isin(cough_type_options) & df["cough_type_4"].isin(cough_type_options)]
print("\ncough_type: ", cough_type_options, len(df))

printValues(df)
#print(sorted(df))
print("\n", df.head(15))