
import pandas as pd
import numpy as np
import streamlit as st
import pickle 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error






# LOGO_URL_LARGE= "C:/Users/Azri/fdm-project/fdm-web-app/logo.png"


# st.image(
#             LOGO_URL_LARGE , width=800
#         )
st.write(""" 
        
         ### classify customers at a *crash risk* and predict *premium coverage* estimate 
          

         """)

#####################################################################################################################################################################
#DECISON TREE MODEL BUILDING#

csv_url = "https://github.com/Ramla24/mm/blob/main/FDM-mini-project-y3s1/fdm-web-app/normalized_data"


 
try:
        # Read the uploaded CSV into a DataFrame
    column_names =['Unnamed: 0','KIDSDRIV','HOMEKIDS','YOJ','INCOME','PARENT1','HOME_VAL','MSTATUS','TRAVTIME','BLUEBOOK','TIF','RED_CAR','OLDCLAIM','CLM_FREQ','REVOKED','MVR_PTS','CLM_AMT','CAR_AGE','CLAIM_FLAG','GEN_AGE','GENDER_M','GENDER_z_F','EDUCATION_<High School','EDUCATION_Bachelors','EDUCATION_Masters','EDUCATION_PhD','EDUCATION_z_High School','OCCUPATION_Clerical','OCCUPATION_Doctor','OCCUPATION_Home Maker','OCCUPATION_Lawyer,OCCUPATION_Manager','OCCUPATION_Professional','OCCUPATION_Student','OCCUPATION_z_Blue Collar','CAR_USE_Commercial','CAR_USE_Private','CAR_TYPE_Minivan','CAR_TYPE_Panel Truck','CAR_TYPE_Pickup','CAR_TYPE_Sports Car','CAR_TYPE_Van','CAR_TYPE_z_SUV','URBANICITY_Highly Urban/ Urban','URBANICITY_z_Highly Rural/ Rural']
    dframe = pd.read_csv(csv_url, encoding='utf-8', names=column_names)
    st.write("CSV successfully loaded!")
except pd.errors.ParserError as e:
    st.error("There was an error parsing the CSV file.")
    st.text(f"Error details: {e}")       

dframe = dframe[dframe['CLAIM_FLAG'] == 1]

dframe = dframe.drop(columns=[ 'Unnamed: 0', 'CLAIM_FLAG'])

x = dframe.drop(columns=['CLM_AMT'])  
y = dframe['CLM_AMT']

y_log= np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_log, test_size=0.3, random_state=42)

basic_tree = DecisionTreeRegressor(random_state=42, max_depth=5)

basic_tree.fit(x_train, y_train)

# y_pred = basic_tree.predict(x_test)


#####################################################################################################################################################################


#********************************************************************************************************************************************************
#SIDEBAR #


st.sidebar.header('User Input Features')

uploaded_csv = st.sidebar.file_uploader("Uplaod your CSV file", type=["csv"])
if uploaded_csv is not None:
    input_df = pd.read_csv(uploaded_csv)
else:
   def user_input_feature():
        male_check = st.sidebar.checkbox("Client is Male")
        female_check = st.sidebar.checkbox("Client is Female")
        GENDER_M = 1 if male_check else 0
        GENDER_z_F = 1 if female_check else 0
     
        GEN_AGE = st.sidebar.slider("Client Age:", value=25, min_value=0, max_value=100, step=1)  # Default to 25

        edu_high_school = st.sidebar.checkbox("Education: <High School")
        edu_bachelors = st.sidebar.checkbox("Education: Bachelors")
        edu_masters = st.sidebar.checkbox("Education: Masters")
        edu_phd = st.sidebar.checkbox("Education: PhD")
        edu_z_high_school = st.sidebar.checkbox("Education: z_High School")
        EDUCATION_High_School = 1 if edu_high_school else 0
        EDUCATION_Bachelors = 1 if edu_bachelors else 0
        EDUCATION_Masters = 1 if edu_masters else 0
        EDUCATION_PhD = 1 if edu_phd else 0
        EDUCATION_z_High_School = 1 if edu_z_high_school else 0


        HOMEKIDS = st.sidebar.slider("How many kids at home:", value=2, min_value=0, max_value=20, step=1)  # Default to 2
        KIDSDRIV = st.sidebar.slider("How many driving age children currently live with user:", value=1, min_value=0, max_value=20, step=1)  # Default to 1



        occ_clerical = st.sidebar.checkbox("Occupation: Clerical")
        occ_doctor = st.sidebar.checkbox("Occupation: Doctor")
        occ_home_maker = st.sidebar.checkbox("Occupation: Home Maker")
        occ_lawyer = st.sidebar.checkbox("Occupation: Lawyer")
        occ_manager = st.sidebar.checkbox("Occupation: Manager")
        occ_professional = st.sidebar.checkbox("Occupation: Professional")
        occ_student = st.sidebar.checkbox("Occupation: Student")
        occ_blue_collar = st.sidebar.checkbox("Occupation: Blue Collar")
        OCCUPATION_Clerical = 1 if occ_clerical else 0
        OCCUPATION_Doctor = 1 if occ_doctor else 0
        OCCUPATION_Home_Maker = 1 if occ_home_maker else 0
        OCCUPATION_Lawyer = 1 if occ_lawyer else 0
        OCCUPATION_Manager = 1 if occ_manager else 0
        OCCUPATION_Professional = 1 if occ_professional else 0
        OCCUPATION_Student = 1 if occ_student else 0
        OCCUPATION_Blue_Collar = 1 if occ_blue_collar else 0



        INCOME = st.sidebar.slider("Yearly Income:", value=50000, min_value=0, step=100, max_value=1000000)  # Default to 50000
        HOME_VAL = st.sidebar.slider("Home value:", value=200000, min_value=0, step=100, max_value=1000000)  # Default to 200000

        # Urbanicity
        urbanicity_urban = st.sidebar.checkbox("Urbanicity: Highly Urban/Urban")
        urbanicity_rural = st.sidebar.checkbox("Urbanicity: z_Highly Rural/Rural")
        URBANICITY_Urban = 1 if urbanicity_urban else 0
        URBANICITY_Rural = 1 if urbanicity_rural else 0

        MSTATUS = st.sidebar.selectbox("Marital Status:", ('Married', 'Unmarried'), index=0)  # Default to 'Married'
        PARENT1 = st.sidebar.selectbox("Is the customer a single parent?", ('Yes', 'No'), index=1)  # Default to 'No'
        YOJ = st.sidebar.slider("Number of years on the current occupation:", value=5, min_value=0, max_value=60)  # Default to 5
        TIF = st.sidebar.slider("Time in force:", value=10, min_value=0, max_value=100)  # Default to 10
        TRAVTIME = st.sidebar.slider("Travel time in hours:", value=1, step=1, min_value=0, max_value=24)  # Default to 1
        BLUEBOOK = st.sidebar.slider("Cost of vehicle (BlueBook):", value=15000, min_value=0, step=100, max_value=1000000)  # Default to 15000
        # Car Use
        car_use_commercial = st.sidebar.checkbox("Car Use: Commercial")
        car_use_private = st.sidebar.checkbox("Car Use: Private")
        CAR_USE_Commercial = 1 if car_use_commercial else 0
        CAR_USE_Private = 1 if car_use_private else 0

        # Car Type
        car_type_minivan = st.sidebar.checkbox("Car Type: Minivan")
        car_type_panel_truck = st.sidebar.checkbox("Car Type: Panel Truck")
        car_type_pickup = st.sidebar.checkbox("Car Type: Pickup")
        car_type_sports_car = st.sidebar.checkbox("Car Type: Sports Car")
        car_type_van = st.sidebar.checkbox("Car Type: Van")
        car_type_suv = st.sidebar.checkbox("Car Type: z_SUV")
        CAR_TYPE_Minivan = 1 if car_type_minivan else 0
        CAR_TYPE_Panel_Truck = 1 if car_type_panel_truck else 0
        CAR_TYPE_Pickup = 1 if car_type_pickup else 0
        CAR_TYPE_Sports_Car = 1 if car_type_sports_car else 0
        CAR_TYPE_Van = 1 if car_type_van else 0
        CAR_TYPE_SUV = 1 if car_type_suv else 0

        CAR_AGE = st.sidebar.slider("Vehicle Age:", value=3, min_value=0, max_value=100)  # Default to 3
        RED_CAR = st.sidebar.selectbox("Is the client's vehicle red?", ('Yes', 'No'), index=1)  # Default to 'No'
        REVOKED = st.sidebar.selectbox("Has the client had their license revoked?", ('Yes', 'No'), index=1)  # Default to 'No'
        OLD_CLAIM = st.sidebar.slider("Previous Claim amount if any:", value=0, min_value=0, max_value=500000)  # Default to 0
        CLM_FREQ = st.sidebar.slider("No. of previous claims:", value=0, min_value=0, max_value=100)  # Default to 0
        MVR_PTS = st.sidebar.slider("MVR points:", value=0, min_value=0, max_value=24)  # Default to 0
        data = {
            'KIDSDRIV': KIDSDRIV,
            'HOMEKIDS': HOMEKIDS,
            'YOJ':  YOJ,
            'INCOME': INCOME,
            'PARENT1':PARENT1, 
            'HOME_VAL': HOME_VAL,
            'MSTATUS': MSTATUS,
            'TRAVTIME': TRAVTIME,
            'BLUEBOOK': BLUEBOOK,
            'TIF':TIF,
            'RED_CAR': RED_CAR,
            'OLDCLAIM': OLD_CLAIM,
            'CLM_FREQ': CLM_FREQ,
            'REVOKED': REVOKED,
            'MVR_PTS': MVR_PTS,
            'CAR_AGE': CAR_AGE,
            'GEN_AGE': GEN_AGE,
            'GENDER_M': GENDER_M,
            'GENDER_z_F': GENDER_z_F,
            'EDUCATION_<High School': EDUCATION_High_School,
            'EDUCATION_Bachelors': EDUCATION_Bachelors,
            'EDUCATION_Masters': EDUCATION_Masters,
            'EDUCATION_PhD': EDUCATION_PhD,
            'EDUCATION_z_High School': EDUCATION_z_High_School,
            'OCCUPATION_Clerical': OCCUPATION_Clerical,
            'OCCUPATION_Doctor': OCCUPATION_Doctor,
            'OCCUPATION_Home Maker': OCCUPATION_Home_Maker,
            'OCCUPATION_Lawyer': OCCUPATION_Lawyer,
            'OCCUPATION_Manager': OCCUPATION_Manager,
            'OCCUPATION_Professional': OCCUPATION_Professional,
            'OCCUPATION_Student': OCCUPATION_Student,
            'OCCUPATION_z_Blue Collar': OCCUPATION_Blue_Collar,
            'CAR_USE_Commercial': CAR_USE_Commercial,
            'CAR_USE_Private': CAR_USE_Private,
            'CAR_TYPE_Minivan': CAR_TYPE_Minivan,
            'CAR_TYPE_Panel Truck': CAR_TYPE_Panel_Truck,
            'CAR_TYPE_Pickup': CAR_TYPE_Pickup,
            'CAR_TYPE_Sports Car': CAR_TYPE_Sports_Car,
            'CAR_TYPE_Van': CAR_TYPE_Van,
            'CAR_TYPE_z_SUV': CAR_TYPE_SUV,
            'URBANICITY_Highly Urban/ Urban': URBANICITY_Urban,
            'URBANICITY_z_Highly Rural/ Rural': URBANICITY_Rural
            
        }
    
        features = pd.DataFrame(data, index=[0])
        return features 
   input_df = user_input_feature()


#*********************************************************************************************************************************************************
#ENCODING INPUT VALUES#

df_raw = pd.read_csv('normalized_data')
df_dropped = df_raw.drop(columns=['Unnamed: 0', 'CLAIM_FLAG', 'CLM_AMT'])
df = pd.concat([input_df, df_dropped] , axis=0)


binary_encode_cols = {
    'MSTATUS': {'Married': 1, 'Unmarried': 0},
    'PARENT1': {'Yes': 1, 'No': 0},
    'RED_CAR': {'Yes': 1, 'No': 0},
    'REVOKED': {'Yes': 1, 'No': 0}
}

for col, mapping in binary_encode_cols.items():
    df[col] = df[col].replace(mapping)


scaler = MinMaxScaler()


df_mm_scaled = (df-df.min())/(df.max()-df.min())

df_mm_scaled = df_mm_scaled.iloc[:1]



#************************************************************************************************************************************************
# Display the features input by the user
st.subheader('User Input features')

if uploaded_csv is not None:
    st.write(df_mm_scaled)
else:
    st.write('Awaiting CSV file to be uploaded using example input parameters')
    st.write(df_mm_scaled)



loaded_classifier_model = pickle.load(open("gaussian.pkl", "rb"))
crash_likeliness = loaded_classifier_model.predict(df_mm_scaled)

claim_probability = loaded_classifier_model.predict_proba(df_mm_scaled)




st.subheader('Prediction')

st.write(crash_likeliness)



if crash_likeliness == 1:


    dframe = pd.read_csv('normalized_data')
    dframe = dframe[dframe['CLAIM_FLAG'] == 1]
    dframe = dframe.drop(columns=['Unnamed: 0', 'CLAIM_FLAG', 'CLM_AMT'])
    dframe = pd.concat([input_df, dframe] , axis=0)
    dframe = dframe.iloc[:1]

    predicted_claim = basic_tree.predict(dframe)

    predicted_claim = np.exp(predicted_claim)
    st.write("""
                ##### :red[Claim Likely]🚩
            """)
    st.write("""
                ##### Confidence:
            """,claim_probability)
    st.write("""
                ### Predicted Claim Amount: $
            """,predicted_claim)
else:
    st.write("""
                #### :green[No claim risk] ✅
            """)
    
    
