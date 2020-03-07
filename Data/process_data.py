import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messges_filepath = Messages csv filepath
    categories_filepath = Cateogries csv filepath
    
    OUTPUT: A merged dataframe of the messages and categories data
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'inner', on = 'id')
    
    return df

def clean_data(df):
    '''
    INPUT:
    df = The merged dataframe from the load_data function
    
    OUTPUT: A cleaned version of df 
    '''
    df = df.drop_duplicates()
    
    categories = df['categories'].str.split(pat=';', n=36, expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    return df


def save_data(df, database_filepath):
    '''
    INPUT:
    df = The cleaned dataframe from the clean_data function
    database_filepath = The filepath of the database
    
    OUTPUT: Save df in the database that is defined
    ''' 
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('DisasterMessages', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()