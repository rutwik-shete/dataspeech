from datasets import load_dataset,Audio
from genderclassification.genderclassify import classifyGender

hindi_dataset = load_dataset("csv",data_files={
    'train': '/Users/rutwikshete/Desktop/Codeing/TeeOff/POC/HindiDataset/train/audio/metadata.csv',
    'test' : '/Users/rutwikshete/Desktop/Codeing/TeeOff/POC/HindiDataset/test/audio/metadata.csv'
})

hindi_dataset = hindi_dataset.map(lambda batch: {'audio': batch['file_name']}, batched=False)

# Step 3: Cast the new 'audio' column to the Audio feature type
hindi_dataset = hindi_dataset.cast_column('audio', Audio())

hindi_dataset = hindi_dataset.map(lambda batch: {'audio': batch['file_name']}, batched=False)

# Step 3: Cast the new 'audio' column to the Audio feature type
hindi_dataset = hindi_dataset.cast_column('audio', Audio())

hindi_dataset_with_gender = hindi_dataset.map(lambda batch: {'gender': classifyGender(batch['file_name'])}, batched=False)

hindi_dataset_with_gender.push_to_hub("RutwikShete/hindi_dataset")