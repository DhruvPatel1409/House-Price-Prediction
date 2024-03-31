import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.pipeline import Pipeline

loaded_model = pickle.load(open('house_model.pkl', 'rb'))
df = pd.read_csv("house cleaned.csv")

def predict_price(input_data):
    return loaded_model.predict(input_data)

def main():
    st.title('House Price Prediction App')

    # Navigation
    with st.sidebar:
        selected = option_menu("Navigate", ['Home', 'Prediction', 'Visualizations'],icons=['bi-house','bi-currency-rupee','bi-crosshair2'])

    if selected == 'Home':
        st.markdown(
            "The house price prediction project utilizes machine learning techniques to analyze various features of a property such as location, size, amenities, and market trends to accurately estimate its market value. By training models on historical housing data, it provides valuable insights for buyers, sellers, and real estate professionals, facilitating informed decision-making in the real estate market."
        )
        st.image('home.jpeg', use_column_width=True)

    elif selected == 'Prediction':
        MSSubClass = st.number_input("MSSubClass")
        MSZoning = st.selectbox("MSZoning", ['Residential Low Density', 'Residential Medium Density', 'Commercial', 'Floating Village Residential', 'Residential High Density'])
        LotArea = st.number_input("LotArea", value=0)
        LotShape = st.selectbox("LotShape", ['Reg', 'Slightly Irreg', 'Moderately Irreg', 'Irregular'])
        LandContour = st.selectbox("LandContour", ['Level', 'Banked', 'Depression', 'HillSide'])
        BldgType = st.selectbox("BldgType", ['1 Family', '2 Family', 'Duplex', 'Townhouse End', 'Townhouse'])
        HouseStyle = st.selectbox("HouseStyle", ['1 Story','2 Story', '1.5 Story Fin.', '1.5 Story Unf.', 'SplitFoyer', 'Second Level', '2.5 story Unf.', '2.5 story Fin.'])
        OverallQual = st.number_input("OverallQual", value=0)
        OverallCond = st.number_input("OverallCond", value=0)
        YearBuilt = st.number_input("YearBuilt", value=0)
        Heating = st.selectbox("Heating", ['Gas forced warm furnace', 'Gas hot water', 'Gravity furnace', 'Wall furnace', 'Others', 'Floor furnace'])
        CentralAir = st.selectbox("CentralAir", ['Yes', 'No'])
        Electrical = st.selectbox("Electrical", ['Std. cir.Brkr.', 'FuseF', 'FuseA', 'FuseP', 'Mixed'])
        GrLivArea = st.number_input("GrLivArea", value=0)
        FullBath = st.number_input("FullBath", value=0)
        BedroomAbvGr = st.number_input("BedroomAbvGr", value=0)
        KitchenAbvGr = st.number_input("KitchenAbvGr", value=0)
        TotRmsAbvGrd = st.number_input("TotRmsAbvGrd", value=0)
        Fireplaces = st.number_input("Fireplaces", value=0)
        GarageType = st.selectbox("GarageType", ['Attached', 'Detached', 'BuiltIn', 'CarPort', 'Basment', '2Types'])
        GarageArea = st.number_input("GarageArea", value=0)
        MoSold = st.number_input("MoSold", value=0)
        YrSold = st.number_input("YrSold", value=0)
        SaleCondition = st.selectbox("SaleCondition", ['Normal', 'Abnormal', 'Partial', 'Adjoining Land', 'Allocation', 'Family'])
        TotalSF = st.number_input("TotalSF", value=0)

        if st.button('Predict'):
            input_df = pd.DataFrame([[MSSubClass, MSZoning, LotArea, LotShape, LandContour, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt, Heating, CentralAir, Electrical, GrLivArea, FullBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageType, GarageArea, MoSold, YrSold, SaleCondition, TotalSF]],
                        columns=['MSSubClass', 'MSZoning', 'LotArea', 'LotShape', 'LandContour', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'Heating', 'CentralAir', 'Electrical', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageType', 'GarageArea', 'MoSold', 'YrSold', 'SaleCondition', 'TotalSF'])

            prediction = predict_price(input_df)
            st.success(f'Predicted Price: {prediction[0]}')

    elif selected == 'Visualizations':
        st.subheader("Analysis Dashboard")
        
        analysis_option = st.selectbox('Select Analysis Option', ['Visualizations', 'Statistical Analysis', 'Data Summary'])

        if analysis_option == 'Visualizations':
    
            bar_column = st.selectbox('Select Column for Bar Graph',['MSZoning', 'LotShape', 'LandContour', 'BldgType',
        'HouseStyle','Heating', 'CentralAir', 'Electrical','GarageType','SaleCondition'])
            
            pie_column = st.selectbox('Select Column for Pie Chart', ['MSZoning', 'LotShape', 'LandContour', 'BldgType',
        'HouseStyle','Heating', 'CentralAir', 'Electrical','GarageType','SaleCondition'])
            
            violin_column = st.selectbox('Select Column for Violin Plot', df.select_dtypes(include=['number']).columns)

            # Color pickers for customization
            bar_color = st.color_picker('Choose Bar Color', '#ff0000')
            pie_color = st.color_picker('Choose Pie Color', '#00FFE0')
            violin_color = st.color_picker('Choose Violin Plot Color', '#E0FF00')

            # Bar graph
            bar_fig = px.bar(df, x=bar_column, color_discrete_sequence=[bar_color], title=f'Bar Graph for {bar_column}')
            st.plotly_chart(bar_fig)

            # Pie chart
            pie_fig = px.pie(df, names=pie_column, color_discrete_sequence=[pie_color], title=f'Pie Chart for {pie_column}')
            st.plotly_chart(pie_fig)

            # Violin plot
            violin_fig = px.violin(df, y=violin_column, box=True, points="all", color_discrete_sequence=[violin_color], title=f'Violin Plot for {violin_column}')
            st.plotly_chart(violin_fig)

        elif analysis_option == 'Statistical Analysis':
            st.subheader('Statistical Analysis')

            st.write("Basic Statistics:")
            st.write(df.describe())

        elif analysis_option == 'Data Summary':
            st.subheader('Data Summary')
            st.write("Number of rows and columns:", df.shape)
            st.write("Columns:", df.columns.tolist())

if __name__ == '__main__':
    main()