import os
import os.path as path
import shutil
import time
from msilib.schema import CreateFolder
import numpy as np
import pandas as pd
from pydub import AudioSegment

start = time.process_time()

metadataFileLocation = "..\\dataset\\kaggle covid-19 cough audio classification\\archive\\metadata_compiled.csv"
archiveFilesLocation = "..\\dataset\\kaggle covid-19 cough audio classification\\archive\\"
selectionFolderName = "selection"
included_extensions = ["ogg","webm"]

df = pd.read_csv(metadataFileLocation)
print("Load csv (s): ", round(time.process_time() - start, 2))

pd.set_option('display.max_columns', None)
#print(df.head(5))

print("Items: ", len(df))

def printValues(dataFrame):
    print("status", dataFrame.status.unique())
    print("quality_1", dataFrame.quality_1.unique())
    print("quality_2", dataFrame.quality_2.unique())
    print("quality_3", dataFrame.quality_3.unique())
    print("quality_4", dataFrame.quality_4.unique())
    print("cough_type_1", dataFrame.cough_type_1.unique())
    print("cough_type_2", dataFrame.cough_type_2.unique())
    print("cough_type_3", dataFrame.cough_type_3.unique())
    print("cough_type_4", dataFrame.cough_type_4.unique())
    print("dyspnea_1", dataFrame.dyspnea_1.unique())
    print("dyspnea_2", dataFrame.dyspnea_2.unique())
    print("dyspnea_3", dataFrame.dyspnea_3.unique())
    print("dyspnea_4", dataFrame.dyspnea_4.unique())
    print("cough_type_1", dataFrame.cough_type_1.unique())
    print("cough_type_2", dataFrame.cough_type_2.unique())
    print("cough_type_3", dataFrame.cough_type_3.unique())
    print("cough_type_4", dataFrame.cough_type_4.unique())
    print("choking_1", dataFrame.choking_1.unique())
    print("choking_2", dataFrame.choking_2.unique())
    print("choking_3", dataFrame.choking_3.unique())
    print("choking_4", dataFrame.choking_4.unique())
    print("congestion_1", dataFrame.congestion_1.unique())
    print("congestion_2", dataFrame.congestion_2.unique())
    print("congestion_3", dataFrame.congestion_3.unique())
    print("congestion_4", dataFrame.congestion_4.unique())
    print("cough_type_1", dataFrame.cough_type_1.unique())
    print("cough_type_2", dataFrame.cough_type_2.unique())
    print("cough_type_3", dataFrame.cough_type_3.unique())
    print("cough_type_4", dataFrame.cough_type_4.unique())
    print("diagnosis_1", dataFrame.diagnosis_1.unique())
    print("diagnosis_2", dataFrame.diagnosis_2.unique())
    print("diagnosis_3", dataFrame.diagnosis_3.unique())
    print("diagnosis_4", dataFrame.diagnosis_4.unique())
    print("nothing_1", dataFrame.nothing_1.unique())
    print("nothing_2", dataFrame.nothing_2.unique())
    print("nothing_3", dataFrame.nothing_3.unique())
    print("nothing_4", dataFrame.nothing_4.unique())
    print("quality_1", dataFrame.quality_1.unique())
    print("quality_2", dataFrame.quality_2.unique())
    print("quality_3", dataFrame.quality_3.unique())
    print("quality_4", dataFrame.quality_4.unique())
    print("respiratory_condition", dataFrame.respiratory_condition.unique())
    print("severity_1", dataFrame.severity_1.unique())
    print("severity_2", dataFrame.severity_2.unique())
    print("severity_3", dataFrame.severity_3.unique())
    print("severity_4", dataFrame.severity_4.unique())
    print("stridor_1", dataFrame.stridor_1.unique())
    print("stridor_2", dataFrame.stridor_2.unique())
    print("stridor_3", dataFrame.stridor_3.unique())
    print("stridor_4", dataFrame.stridor_4.unique())
    print("wheezing_1", dataFrame.wheezing_1.unique())
    print("wheezing_2", dataFrame.wheezing_2.unique())
    print("wheezing_3", dataFrame.wheezing_3.unique())
    print("wheezing_4", dataFrame.wheezing_4.unique())

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

#printValues(df)
#print(sorted(df))
#print("\n", df.head(15))

# Extract files
randomRecordings = df["uuid"].sample(200)

# Create folder for selection
try:
    shutil.rmtree(archiveFilesLocation + selectionFolderName)
except OSError as e:
    print("Error: %s : %s" % (archiveFilesLocation + selectionFolderName, e.strerror))

if (not path.exists(archiveFilesLocation + selectionFolderName)):
    os.makedirs(archiveFilesLocation + selectionFolderName)

archive_file_list = [fn for fn in os.listdir(archiveFilesLocation)
                        if any(fn.endswith(ext) for ext in included_extensions)]

def convertFile(recordingIdStr):
    file_name = [fn for fn in archive_file_list if fn.startswith(recordingIdStr)][0]
    file_path = archiveFilesLocation + file_name
    if (not path.exists(file_path)):
        raise Exception("Missing file from archive: ", file_path)
    try:
        #shutil.copy2(file_path, archiveFilesLocation+selectionFolderName)
        sound = AudioSegment.from_file(file_path)
        copied_file = sound.export(archiveFilesLocation+selectionFolderName + "\\" + recordingIdStr + ".wav", format="wav").file_name
    except:
        print("Convert failed: ", file_path)
    else:
        print(copied_file)

for recordingId in randomRecordings:
    convertFile(recordingId)

print("Converted # files: ", len(randomRecordings))
print("Runtime (s): ", round(time.process_time() - start, 2))