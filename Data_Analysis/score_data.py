import pymysql
import pandas as pd
import numpy as np

import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report





class AWSUploader():
    def __init__(self):
        load_dotenv()
        hostname = os.getenv('seame.cp2yk2eew78f.eu-central-1.rds.amazonaws.com')
        username = os.getenv('root')
        password = os.getenv('12341234')
        database = os.getenv('seame')

        self.connection = pymysql.connect(
            host='seame.cp2yk2eew78f.eu-central-1.rds.amazonaws.com',
            port=3306,
            user='root',
            passwd='12341234',
            db='seame',
        )
        self.cursor = self.connection.cursor()
        self.is_mysql_connected(self.connection)
    
    def is_mysql_connected(self, connection):
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                print("success to connect to MySQL")
                return True
        except pymysql.MySQLError as e:
            print(f"Failed to connect to MySQL: {e}")
            return False
    
    def upload_csv(self, file_path):
        data = pd.read_csv(file_path, index_col=0)
        print(data)
        columns = ['Time', 'Throttle', 'Speed', 'Steering', 'Brake', 'Acc', 'LocationX', 'LocationY', 'Distance','Fuel_efficiency']
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO raw_data ({', '.join(columns)}) VALUES ({placeholders})"
        # Convert DataFrame to list of tuples
        data_tuples = [tuple(x) for x in data.to_numpy()]
        try:
            self.cursor.executemany(sql, data_tuples)
            self.connection.commit()
            print("All data uploaded successfully")
        except pymysql.MySQLError as e:
            print(f"Error during bulk insert: {e}")
            self.connection.rollback()
            
            
    def upload_dataframe(self, df):
        print(df)
        columns = ['Time', 'Throttle', 'Speed', 'Steering', 'Brake', 'Acc', 'LocationX', 'LocationY', 'Distance','Fuel_efficiency']
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO raw_data ({', '.join(columns)}) VALUES ({placeholders})"
        # Convert DataFrame to list of tuples
        data_tuples = [tuple(x) for x in df.to_numpy()]
        try:
            self.cursor.executemany(sql, data_tuples)
            self.connection.commit()
            print("All data uploaded successfully")
        except pymysql.MySQLError as e:
            print(f"Error during bulk insert: {e}")
            self.connection.rollback()
            
    
    def upload_preprocess_data(self, data_row):
        columns = ['Time', 'Distance', 'Fuel_efficiency', 'Acc_count', 'Dec_count', 'Night_time']
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO pp_data ({', '.join(columns)}) VALUES ({placeholders})"
        try:
            self.cursor.execute(sql, data_row)
            self.connection.commit()
            print("Preprocessing_Data uploaded successfully")
        except pymysql.MySQLError as e:
            print(f"Error during insert: {e}")
            self.connection.rollback()
            
    def upload_score(self, data_row):
        columns = ['score_safety', 'score_fuel', 'score_time', 'Distance', 'battery', 'Distance_available','score_Acc','score_Dec','score_night','driving_type']
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO result_data ({', '.join(columns)}) VALUES ({placeholders})"
        try:
            self.cursor.execute(sql, data_row)
            self.connection.commit()
            print("Score_Data uploaded successfully")
        except pymysql.MySQLError as e:
            print(f"Error during insert: {e}")
            self.connection.rollback()

            

    def truncate_table(self):
        try:
            self.cursor.execute("TRUNCATE TABLE raw_data")
            self.connection.commit()
            print("All data in 'raw_data' has been deleted successfully.")
        except pymysql.MySQLError as e:
            print(f"Error while truncating table: {e}")
            self.connection.rollback()

    def truncate_score(self):
        try:
            self.cursor.execute("TRUNCATE TABLE result_data")
            self.connection.commit()
            print("All data in 'result_data' has been deleted successfully.")
        except pymysql.MySQLError as e:
            print(f"Error while truncating table: {e}")
            self.connection.rollback()
    
    def fetch_data(self):
        try:
            # Define your SQL query to select data
            sql = "SELECT * FROM pp_data"
            
            # Execute the SQL query
            self.cursor.execute(sql)
            
            # Fetch all the rows from the result set
            rows = self.cursor.fetchall()
            
            # If rows exist, convert them to a DataFrame
            if rows:
                # Get column names
                columns = [col[0] for col in self.cursor.description]
                # Convert rows to DataFrame
                df = pd.DataFrame(rows, columns=columns)
                print("Data fetched successfully.")
                return df
            else:
                print("No data found in the table.")
                return None
        except pymysql.MySQLError as e:
            print(f"Error while fetching data: {e}")   
            
class ModelTrainer:
    def __init__(self):
        self.model = None        
             
    def calculate_safety_score(self,z_score): 
        # Z-Score가 높을수록 낮은 점수
        if z_score < -1.28:
            return 100  
        elif z_score < -0.84:
            return 85
        elif z_score < 0.25:
            return 70
        elif z_score < 0.84:
            return 55
        else:
            return 40
    
    def calculate_fuel_score(self,z_score): # fuel=전체 거리/ 쓰로틀 
        # Z-Score가 높을수록 높은 점수
        if z_score > 1.28:      # 낮은 점수 (안전)  상위10%
            return 100           
        elif z_score >= 0.84:   # 상위 20 %                              
            return 85
        elif z_score >= -0.25:  # 중간 점수 (보통) 40%
            return 70
        elif z_score >= -0.84:
            return 55
        else:                   # 높은 점수 (위험)
            return 40
    

    def calculate_time_score(self,z_score): #time_score = time/distance 클수록 안좋음
        # Z-Score가 높을수록 낮은 점수
        if z_score < -1.28:
            return 100  
        elif z_score < -0.84:
            return 85
        elif z_score < 0.25:
            return 70
        elif z_score < 0.84:
            return 55
        else:
            return 40
        
        
        
    def data_train(self,X,Y,average):
        # test&train data split 
        x = X
        y = Y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # model learning
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(x_train, y_train)

        # model predict
        y_pred = self.model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'mean:{average}')
        print(f'Mean Squared Error: {mse}')
        print(f'R² Score: {r2}')

    def predict_new_data(self, new_data):
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        prediction = self.model.predict(new_data)
        return 
    
    def classify_score(self,score):
        return '2' if score >= 70 else '1'
    
    # def classify_type(self,x,y):
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    #     # decision tree model
    #     tree_model = DecisionTreeClassifier()
    #     tree_model.fit(x_train, y_train)
    #     y_pred = tree_model.predict(x_test)
    #     tree_accuracy = accuracy_score(y_test, y_pred)
    #     tree_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    #     print(f'Decision Tree Accuracy: {tree_accuracy}')
    #     print(f'Decision Tree Classification Report:\n{tree_report}')

        
def main():
    uploader = AWSUploader()
    uploader_trainer = ModelTrainer()
    
    # fetch_data 
    result=uploader.fetch_data() #fetch data from database
    result['Distance'] = result['Distance']/1000 # m_2_km

    distance = result['Distance']
    over_cnt = result['Acc_count'] + result['Dec_count'] #all_count
    
    
    #result.iloc[:-1][['Distance', 'Acc']]
    #### score_safety
    print('================= score_safety =================')
    result['over_cnt'] = result['Acc_count'] + result['Dec_count'] #all_count
    result['ratio_over_dis']=(result['Acc_count'] + result['Dec_count'])/result['Distance'] #km당 가속 횟수
    mean_safety  = result['ratio_over_dis'].mean() #result.iloc[:-1]['ratio_over_dis'].mean()
    std_safety = result['ratio_over_dis'].std() 
    # Z-Score 계산
    result['z_score_safty'] = (result['ratio_over_dis'] - mean_safety) / std_safety
    # 점수 측정
    result['score_safety'] = result['z_score_safty'].apply(uploader_trainer.calculate_safety_score)
    print(result.tail(5))
    # data train
    uploader_trainer.data_train(result[['Acc_count', 'Dec_count', 'Distance', 'ratio_over_dis', 'z_score_safty']],result['score_safety'],mean_safety)
    
    # # new data
    # new_data = pd.DataFrame([result.iloc[-1][['Acc_count', 'Dec_count', 'Distance', 'ratio_over_dis', 'z_score_safty']]])
    # predicted_value = uploader_trainer.predict_new_data(new_data)
    # result.iloc[-1]['score_safety']=predicted_value[0]
    # print(predicted_value[0])
    
    # safety_detail
    mean_Acc = (result['Acc_count']/result['Distance']).mean()
    std_Acc  = (result['Acc_count']/result['Distance']).std()
    mean_Dec = (result['Dec_count']/result['Distance']).mean()
    std_Dec  = (result['Dec_count']/result['Distance']).std()
    
    result['Z_score_Acc'] = (result['Acc_count']/result['Distance']-mean_Acc) / std_Acc
    result['Z_score_Dec'] = (result['Dec_count']/result['Distance']-mean_Dec) / std_Dec

    result['score_Acc'] = result['Z_score_Acc'].apply(uploader_trainer.calculate_safety_score)
    result['score_Dec'] = result['Z_score_Dec'].apply(uploader_trainer.calculate_safety_score)
    result['score_night'] = (result['Time']-result['Night_time'])*100/result['Time']
    
    uploader_trainer.data_train(result[['Acc_count', 'Dec_count', 'Distance', 'Z_score_Acc']],result['score_Acc'],mean_Acc)
    uploader_trainer.data_train(result[['Acc_count', 'Dec_count', 'Distance', 'Z_score_Dec']],result['score_Dec'],mean_Dec)



    #### score_efficiency 
    print('================= score_efficiency =================')
    mean_efficiency  = result['Fuel_efficiency'].mean()
    std_efficiency = result['Fuel_efficiency'].std() 
    # Z-Score 계산
    result['z_score_efficiency'] = (result['Fuel_efficiency'] - mean_efficiency) / std_efficiency
    # 점수 측정
    result['score_efficiency'] = result['z_score_efficiency'].apply(uploader_trainer.calculate_fuel_score)
    print(result[['Time', 'Distance', 'Fuel_efficiency', 'z_score_efficiency','score_efficiency']].tail(5))
    # data train
    uploader_trainer.data_train(result[['Time', 'Distance', 'Fuel_efficiency', 'z_score_efficiency']],result['score_efficiency'],mean_efficiency)



    #### score_time
    print('================= score_time =================')
    result['ratio_time_dis']=result['Time']/result['Distance'] #km당 걸린 주행 시간
    mean_time  = result['ratio_time_dis'].mean()
    std_time = result['ratio_time_dis'].std() 
    # Z-Score 계산
    result['z_score_time'] = (result['ratio_time_dis'] - mean_time) / std_time
    # 점수 측정
    result['score_time'] = result['z_score_time'].apply(uploader_trainer.calculate_time_score)
    print(result[['Time', 'Distance', 'ratio_time_dis', 'z_score_time','score_time']].tail(20))
    # 점수 측정
    uploader_trainer.data_train(result[['Time', 'Distance', 'ratio_time_dis', 'z_score_time']],result['score_time'],mean_time)


    ## Classification driving type
    result['C_safety'] = result['score_safety'].apply(uploader_trainer.classify_score)
    result['C_efficiency'] = result['score_efficiency'].apply(uploader_trainer.classify_score)
    result['C_time']= result['score_time'].apply(uploader_trainer.classify_score)
    
    result['C_total'] = result['C_safety']+result['C_efficiency']+result['C_time']

    # 레이블 인코딩
    label_encoder = LabelEncoder()
    result['label_C_total'] = label_encoder.fit_transform(result['C_total'])
    print(result[['C_safety','C_efficiency','C_time','C_total','label_C_total']].tail(5))





    
    
    
    
    
    
    
    

    
    
    
    #시각화
    plt.figure(figsize=(10,4))
    plt.plot(distance, over_cnt, marker='o',linestyle='None', color='green',markersize=5, label='Value over Distance')
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x // 3600):02d}:{int((x % 3600) // 60):02d}:{int(x % 60):02d}'))
    plt.title("Count of Acc/Dec")
    plt.xlabel("Distance(km)")
    plt.ylabel("Count")
    # plt.ylim(-30,30)
    plt.show()
    
    # Score_data
    # socre_decimal_point=1
    score_safety = result['score_safety'].iloc[-1]    
    score_fuel = result['score_efficiency'].iloc[-1]    
    score_time = result['score_time'].iloc[-1]   
    Distance = result['Distance'].sum()                        #마지막줄 driving_data.loc['Distance'].astype(int)
    battery = ((474-Distance)/474)*100
    print(Distance,battery)
    Distance_available = 474-Distance # 연비402~547km, 평균 474
    score_Acc = result['score_Acc'].iloc[-1] 
    score_Dec = result['score_Dec'].iloc[-1] 
    score_night = result['score_night'].iloc[-1]
    driving_type = result['C_total'].iloc[-1] #0~8


    socre_data=(score_safety, score_fuel, score_time, Distance, battery, Distance_available,score_Acc,score_Dec,score_night,driving_type)
    
    # truncate_score data
    uploader.truncate_score()
    uploader.upload_score(socre_data)
    
if __name__ == '__main__':
    main()