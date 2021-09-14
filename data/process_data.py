import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data from the raw data files.

    Args:
        messages_filepath (str): path of raw data containing disaster messages
        categories_filepath (str): path of raw data containing catogery of the
                             messages

    Returns:
        df (pd.DataFrame): merged database (messages and categories)
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    '''
    Clean data of the merged DataFrame e.g. splitting out categories into
    several one-hot encoded columns.

    Args:
        df (pd.DataFrame): merged database

    Returns:
        df (pd.DataFrame): merged and cleaned database
    '''

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe and use this row to
    # extract a list of new column names for categories.
    row = list(categories.iloc[0,])
    category_colnames = [col[:-2] for col in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values into 0 or 1
    for column in category_colnames:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        # Some columns have values greater than one
        # e.g. in the column "related", there are rows with value = 2.
        # Overwrite these cases by setting value = 1.
        categories.loc[categories[column] > 1, column] = 1

    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates(subset='id')

    return df

def save_data(df, database_filename):
    '''
    Save data into the SQLite database with the table name "messages".

    Args:
        df (pd.DataFrame): database
        database_filename (str): database name
    '''
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')

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
