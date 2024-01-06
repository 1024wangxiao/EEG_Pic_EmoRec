import pandas as pd

input_filename = r"D:\Postgraduate\SHU-AI\Professor_Wang\Projects\EEG_Multimodal_Project\code\EEG_Pic_EmoRec\Data\text_continue.xlsx"
df = pd.read_excel(input_filename, engine='openpyxl', header=None)

print(df)