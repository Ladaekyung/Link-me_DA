import pymysql
import pandas as pd
import numpy as np

import os
from dotenv import load_dotenv

import sklearn
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

class AWSUploader():
    def __init__(self):
        load_dotenv()
        hostname = os.getenv('seame.cp2yk2eew78f.eu-central-1.rds.amazonaws.com')  #seame.c0aj97hjupda.eu-central-1.rds.amazonaws.com
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
        #print(df)
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

    def upload_overpoint(self, df):
        #print(df)
        columns = ['x_over', 'y_point','type']
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO overpoint_data ({', '.join(columns)}) VALUES ({placeholders})"
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
        
            
    def truncate_table(self):
        try:
            self.cursor.execute("TRUNCATE TABLE raw_data")
            self.connection.commit()
            print("All data in 'raw_data' has been deleted successfully.")
        except pymysql.MySQLError as e:
            print(f"Error while truncating table: {e}")
            self.connection.rollback()
            
    def truncate_overpoint(self):
        try:
            self.cursor.execute("TRUNCATE TABLE overpoint_data")
            self.connection.commit()
            print("All data in 'overpoint_data' has been deleted successfully.")
        except pymysql.MySQLError as e:
            print(f"Error while truncating table: {e}")
            self.connection.rollback()
    
    def fetch_data(self):
        try:
            # Define your SQL query to select data
            sql = "SELECT * FROM raw_data"
            
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

    

        
def main():
    uploader = AWSUploader()
    
    # read simulation data
    df=pd.read_csv("C:/Python_La/Seame_FBS/Normal/normal1.csv") #good 1-30만 넣었음/ good2, /bad1-2 #bad2/bad1-2 #normal1~5

    # delete before starting 
    df=df.loc[df['Distance']>0]  
    df=df.iloc[:-1,1:] #마지막에 데이터, index column 한줄씩 지우기
    
    # group data and calculate the mean    (6행->1행 0.1초 단위 변경)
    batch_size=6 
    df = df.groupby(np.arange(len(df))//batch_size).mean()
    
    # upload raw data
    uploader.truncate_table()                            # deleting all datas from table
    uploader.upload_dataframe(df)                             # now all datas in table
    
    # fetch raw data 
    result=uploader.fetch_data() #fetch data from database
    
    # Change time unit(second_2_time)
    decimal_point=1
    result['Time']=result['Time'].round(decimal_point)
    result['Time']=result['Time']+36000-2340  # control time(to check night drive)
    result['date_Time'] = pd.to_timedelta(result['Time'], unit='s')
    
    # Discriminant night driving    
    Night_time_start= '23:00:00' #23~05시
    Night_time_end = '05:00:00'
    result['is_night'] = ((result['date_Time'] >= Night_time_start) | (result['date_Time']<= Night_time_end))
    
    # night driving time index(starting,end) 
    if not result['is_night'].any():
        start_night_drive = 0
        end_night_drive = 0
    else:          
        start_night_drive = result['is_night'].idxmax()
        end_night_drive = result['is_night'].last_valid_index()
    print(start_night_drive,end_night_drive,'*******************')
    
    night_coordinate = result[result['is_night'] == True][['LocationX','LocationY']]
    night_coordinate['type'] = 2    
    r_night_driving_time = round(result.loc[end_night_drive,'Time']-result.loc[start_night_drive,'Time'],decimal_point)
    
    
    
    # group data and calculate the mean   (0.5 초 단위 변경)
    batch_size2=30/batch_size 
    result = result.groupby(np.arange(len(df))//batch_size2).mean()
    
    # Distance
    result['Distance']=result['Distance'].round(decimal_point)
    
    #가속, /감속
    # print(result.tail(5))
    result['Acc']=result['Acc'].rolling(window=3).mean()
    print(result.tail(5))
    over_acc=10
    under_dec=-13
    result['is_acc']=result['Acc']>=over_acc   #+10
    result['is_dec']=result['Acc']<=under_dec  #
    print(result.tail(5))
    
    # 가속/감속 
    # standard 
    y_acc_standard = np.full_like(result['Time'], over_acc)
    y_dec_standard = np.full_like(result['Time'], under_dec)
    # Overpoint Acc 
    y_acc = np.where(result['is_acc'], result['Acc'], None)
    y_dec = np.where(result['is_dec'], result['Acc'], None) 
    
    # Overpoint Coordinate
    acc_coordinate = result[result['is_acc'] == True][['LocationX','LocationY']]
    dec_coordinate = result[result['is_dec'] == True][['LocationX','LocationY']]
    acc_coordinate['type'] = 0
    dec_coordinate['type'] = 1
    over_table = pd.concat([acc_coordinate, dec_coordinate,night_coordinate])
    print(over_table)
    
    # Preprocess_Driving data
    print('==========Result==========')
    r_time = round(result['Time'].iloc[-1]-result['Time'].iloc[0],decimal_point)
    r_distance = result['Distance'].iloc[-1]
    r_fuel_efficency = r_distance/result['Throttle'].sum()   # 전체 거리/ 쓰로틀
    r_acc_cnt = result['is_acc'].sum()
    r_dec_cnt = result['is_dec'].sum()
    
    
    print("Time(s):",r_time) # Time_second
    print("Time(hh/mm/ss):",result['date_Time'].iloc[-1]-result['date_Time'].iloc[0]) #Time_hh/mm/ss
    print("Distance(m):",r_distance)
    print("Fuel_efficiency:",r_fuel_efficency)
    print("Acc count:",r_acc_cnt)
    print("Dec count:",r_dec_cnt)
    print("Night driving time:",r_night_driving_time)
    print('==========================\n')
    print("Mean Speed:",round(result['Speed'].mean(),3))
    data_result=(r_time, r_distance, r_fuel_efficency, r_acc_cnt, r_dec_cnt, r_night_driving_time)
    #uploader.upload_preprocess_data(data_result)
    
    # upload overpoint data
    uploader.truncate_overpoint()
    uploader.upload_overpoint(over_table)
    
    #시각화
    plt.figure(figsize=(10,4))
    plt.plot(result['Time'], result['Acc'], label='Value over Time')
    plt.plot(result['Time'],y_acc_standard,"r--")
    plt.plot(result['Time'],y_dec_standard,"r--")
    plt.plot(result['Time'],y_acc,marker='o',linestyle='None',color='red',markersize=5)
    plt.plot(result['Time'],y_dec,marker='o',linestyle='None',color='blue',markersize=5)
    y_night=np.where(result['is_night'], result['Acc'], None) 
    plt.plot(result['Time'],y_night,color='purple')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x // 3600):02d}:{int((x % 3600) // 60):02d}:{int(x % 60):02d}'))
    plt.title("Checking Over Acc")
    plt.xlabel("Time (hh:mm:ss)")
    plt.ylabel(r'ACC [km/h$^2$]')
    plt.ylim(-30,30)
    plt.show()


if __name__ == '__main__':
    main()
    
    # print(result.head(5))    
    # DataFrame 주요 속성
    # print('=====데이터 특성=====\n')
    # print('-Shape: ', df.shape)
    # print('-Index: ', df.index)
    # print('-Columns: ', df.columns)
    # print('-DTypes:\n', df.dtypes)
    # print('-Values:\n', df.values)