import pymysql
import pandas as pd
import numpy as np

import os
from dotenv import load_dotenv


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
            
            
    def upload_preprocess_data(self, file_path):
        data = pd.read_csv(file_path, index_col=0)
        # print(data.dtypes())
        columns = ['Time', 'Distance', 'Fuel_efficiency', 'Acc_count', 'Dec_count', 'Night_time']
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO pp_data ({', '.join(columns)}) VALUES ({placeholders})"
        # Convert DataFrame to list of tuples
        data_tuples = [tuple(x) for x in data.to_numpy()]
        try:
            self.cursor.executemany(sql, data_tuples)
            self.connection.commit()
            print("All data uploaded successfully")
        except pymysql.MySQLError as e:
            print(f"Error during bulk insert: {e}")
            self.connection.rollback()
    

            
    def truncate_table(self):
        try:
            self.cursor.execute("TRUNCATE TABLE raw_data")
            self.connection.commit()
            print("All data in 'raw_data' has been deleted successfully.")
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
    df=pd.read_csv("C:/Python_La/Seame_FBS/Good/good1.csv")

    # delete before starting 
    df=df.loc[df['Distance']>0]  
    df=df.iloc[:-1,1:] #마지막에 데이터, index column 한줄씩 지우기
    
    # group data and calculate the mean(한 순간 을 나누려고) 6행->0.1초
    batch_size=30 
    num_batches= len(df) // batch_size
    df = df.groupby(np.arange(len(df))//batch_size).mean()
    #result = df.groupby(df.index // batch_size).apply(lambda x: x.mean())
    
    # upload data
    uploader.truncate_table() # deleting all datas from table
    uploader.upload_dataframe(df) # now all datas in table
    #uploader.upload_csv("C:/Python_La/Seame_FBS/Good/good1.csv") # now all datas in table
    
if __name__ == '__main__':
    main()