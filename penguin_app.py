import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the DataFrame
csv_file = 'D:\Programs\Python\General\Streamlit\penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})

# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression(max_iter = 1000)
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

st.sidebar.title("Penguin Species Prediction App")

islandval = st.sidebar.selectbox("Island", ('Bisoce', 'Dream', 'Torgersen'))
bill_length_mm_val = st.sidebar.slider("Bill Length (mm)", float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
bill_depth_mm_val = st.sidebar.slider("Bill Depth (mm)", float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
flipper_length_mm_val = st.sidebar.slider("Flipper Length (mm)", float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
body_mass_g_val = st.sidebar.slider("Body Mass", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))
sex_val = st.sidebar.selectbox("Sex", ('Male', 'Female'))

classifier = st.sidebar.selectbox("Classifier", ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

if sex_val == 'Male':
	sex_numeric_val = 0
else:
	sex_numeric_val = 1

if islandval == 'Biscoe':
	island_numeric_val = 0
elif islandval == 'Dream':
	island_numeric_val = 1
else:
	island_numeric_val = 2

@st.cache()
def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    species = model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
    species = species[0]
    if species == 0:
        return "Adelie"
    elif species == 1:
        return "Chinstrap"
    else:
        return "Gentoo"

if st.sidebar.button("Predict"):
	if classifier =='Support Vector Machine':
		species_type = prediction(svc_model, island_numeric_val, bill_length_mm_val, bill_depth_mm_val, flipper_length_mm_val, body_mass_g_val, sex_numeric_val)
		score = svc_model.score(X_train, y_train)
	
	elif classifier =='Logistic Regression':
		species_type = prediction(log_reg, island_numeric_val, bill_length_mm_val, bill_depth_mm_val, flipper_length_mm_val, body_mass_g_val, sex_numeric_val)
		score = log_reg.score(X_train, y_train)
	
	else:
		species_type = prediction(rf_clf, island_numeric_val, bill_length_mm_val, bill_depth_mm_val, flipper_length_mm_val, body_mass_g_val, sex_numeric_val)
		score = rf_clf.score(X_train, y_train)
	
	st.write("Species predicted:", species_type)
	st.write("Accuracy score of this model is:", score)
