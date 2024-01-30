import os
import shutil
from PIL import Image
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mplfinance as mpf


# Function to generate candlestick images for stock market data
def generate_images(df, target_folder, non_target_folder, last_n_bar_folder, bar_length=10):
    # Ensure required directories exist
    for folder in [target_folder, non_target_folder, last_n_bar_folder]:
        os.makedirs(folder, exist_ok=True)  # Create the directory if it doesn't exist
    # Removing the last row from the dataframe
    df = df.drop(df.tail(1).index)
    # Looping through the dataframe to create images for each specified length of bars
    for i in range(len(df) - bar_length + 1):
        # Extracting a subset of the dataframe for the current window
        ohlc_data = df.iloc[i:i + bar_length]
        # Extracting last three bars from the window to check for a specific pattern
        data = ohlc_data.iloc[bar_length - 3:bar_length]
        open1, open2, open3 = data['Open'].values
        high1, high2, high3 = data['High'].values
        low1, low2, low3 = data['Low'].values
        close1, close2, close3 = data['Close'].values
        # Determine if the current sequence of bars matches the specified trading pattern:
        # 1. Confirm each bar's closing price is higher than its opening price, indicating an upward trend for all three bars.
        #    This is checked by comparing the closing and opening prices of each bar in a consolidated manner using the `all` function.
        # 2. Ensure the closing prices are in ascending order across the three bars. This checks for a consistent increase in value,
        #    reinforcing the upward trend.
        # 3. Ascertain that the highs and lows of the bars are also in ascending order. This indicates a rising market without sudden drops.
        # 4. Verify that for each bar, the sum of the lengths of the upper and lower shadows (the lines extending from the body of the candlestick)
        #    is smaller than the body's length (the difference between the opening and closing prices). This condition suggests that the market
        #    sentiment is strongly in favor of the trend, with less price volatility during each period.
        # 5. Finally, check if the closing price of the third bar is greater than the highest high of the previous bars in the sequence.
        #    This confirms that the last bar not only follows the trend but also achieves a new high compared to the preceding bars,
        #    potentially indicating a strong bullish momentum.
        is_my_pattern = (
            # Check if closing prices are greater than opening prices for all three bars
                all(close > open for close, open in zip([close1, close2, close3], [open1, open2, open3])) and

                # Ensure closing prices are in ascending order across the three bars
                close1 < close2 < close3 and

                # Verify that the highs and lows are in ascending order
                high1 < high2 < high3 and
                low1 < low2 < low3 and

                # Check if the upper and lower shadows are smaller than the bodies for each bar
                all(((high - close) + (open - low)) < (close - open)
                    for high, close, open, low in
                    zip([high1, high2, high3], [close1, close2, close3], [open1, open2, open3], [low1, low2, low3])) and

                # Ensure the closing price of the third bar is greater than the highest high of the previous bars
                close3 > max(ohlc_data.iloc[:bar_length - 1]['High'])
        )

        folder_name = target_folder if is_my_pattern else non_target_folder

        if i == len(df) - bar_length:
            folder_name = last_n_bar_folder

        mpf.plot(
            ohlc_data,
            type='candle',
            style='yahoo',
            axisoff=True,
            savefig=dict(
                fname=f'./{folder_name}/{ticker}{"_ohlc_target_" if is_my_pattern else "_ohlc_"}{i + 1}.png',
                bbox_inches='tight'
            )
        )


last_n_bar_folder_name = 'classification_imgs/last_n_bar_imgs'
pre_target_folder_name = 'classification_imgs/pre_target_imgs'
target_folder_name = 'classification_imgs/target_imgs'
non_target_folder_name = 'classification_imgs/non_target_imgs'
bar_length = 10
tickers = [
    'BTC-USD',
    'ETH-USD',
    'DOGE-USD',
    'BNB-USD',
    'SOL-USD',
    'XRP-USD',
    'INJ-USD',
    'LINK-USD',
    'ADA-USD',
    'AVAX-USD'
]
start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
interval = '1d'

for ticker in tickers:
    df = yf.download(
        tickers=ticker,
        start=start_date,
        interval=interval,
        progress=False
    )[['Open', 'High', 'Low', 'Close']]
    generate_images(
        df,
        target_folder_name,
        non_target_folder_name,
        last_n_bar_folder_name,
        bar_length
    )


def organize_images(tickers, target_folder, non_target_folder, pre_target_folder):
    # This function organizes images into their respective folders based on whether they match the target pattern.
    # It processes images in the target folder and re-categorizes them if needed, based on the associated stock ticker.
    # Ensure required directories exist
    for folder in [target_folder, non_target_folder, pre_target_folder]:
        os.makedirs(folder, exist_ok=True)
    # Iterate over each image file in the target folder.
    for filename in os.listdir(target_folder):
        # Iterate through each ticker to find a matching stock in the filename.
        for ticker in tickers:
            # Split the filename to extract relevant parts.
            parts = filename.split('_')
            # Check if the current ticker matches the first part of the filename.
            if ticker == parts[0]:
                # Extract the index from the filename, assuming the format is '{ticker}_ohlc_{index}.png'.
                index = int(parts[-1].split('.')[0])

                # Define filenames for previous and new target images.
                prev_filename = f'{ticker}_ohlc_{index - 3}.png'
                new_target_filename = f'{ticker}_ohlc_{index}.png'
                new_target_path = os.path.join(target_folder, new_target_filename)

                # Rename the file if it doesn't already exist in the target folder.
                if not os.path.exists(new_target_path):
                    # If a corresponding previous file exists in the non-target folder, rename it.
                    if os.path.exists(os.path.join(non_target_folder, prev_filename)):
                        os.rename(
                            os.path.join(non_target_folder, prev_filename),
                            os.path.join(non_target_folder, f'{ticker}_ohlc_target_{index - 3}.png')
                        )
                    # Rename the current file in the target folder to the new target filename.
                    os.rename(
                        os.path.join(target_folder, filename),
                        new_target_path
                    )

    # Move files from the non-target folder to the pre-target folder if they are marked as 'target'.
    for filename in os.listdir(non_target_folder):
        # Check if the filename contains the word 'target'.
        if "target" in filename:
            # Define the source and destination paths.
            source_path = os.path.join(non_target_folder, filename)
            destination_path = os.path.join(pre_target_folder, filename)
            # Move the file from the non-target folder to the pre-target folder.
            shutil.move(source_path, destination_path)


# Call the function to organize images.
organize_images(tickers, target_folder_name, non_target_folder_name, pre_target_folder_name)


def select_and_remove_files(pre_target_folder, non_target_folder):
    # This function equalizes the number of files in the non-target folder to match the count in the pre-target folder.
    # This is done by randomly selecting a subset of files in the non-target folder and removing the rest.

    # Count the number of files in the pre-target folder.
    pre_target_count = len(os.listdir(pre_target_folder))

    # List all files in the non-target folder.
    non_target_files = os.listdir(non_target_folder)

    # Randomly select a number of files from the non-target folder equal to the count in the pre-target folder.
    selected_non_target_files = random.sample(non_target_files, pre_target_count)

    # Iterate over all files in the non-target folder.
    for file_name in non_target_files:
        # If the file is not in the selected subset, remove it.
        if file_name not in selected_non_target_files:
            file_path = os.path.join(non_target_folder, file_name)
            os.remove(file_path)


# Call the function to select and remove excess files.
select_and_remove_files(pre_target_folder_name, non_target_folder_name)


def resize_images(folder, width=128, height=128):
    # This function resizes all .png images in the specified folder to the given dimensions.
    # This is typically used to standardize the size of images for consistent processing and analysis.

    # Iterate over all files in the given folder.
    for fn in os.listdir(folder):
        # Check if the file is a .png image.
        if fn.endswith('.png'):
            # Construct the full path to the image file.
            image_path = os.path.join(folder, fn)
            # Open the image.
            image = Image.open(image_path)
            # Resize the image to the specified width and height.
            resized_image = image.resize((width, height))
            # Construct the path where the resized image will be saved.
            resized_image_path = os.path.join(folder, fn)
            # Save the resized image back to the folder, overwriting the original.
            resized_image.save(resized_image_path)

# Call the function to resize images in specific folders.


folders = [pre_target_folder_name, non_target_folder_name, last_n_bar_folder_name]
for folder in folders:
    resize_images(folder)


def predict_target_images(non_target_folder, pre_target_folder, last_n_bar_folder):
    # This function trains a logistic regression model to predict whether images represent a target pattern.
    # It uses images from the non-target and pre-target folders for training and testing,
    # and then applies the model to new images in the last_n_bar_folder.

    # List and filter all .png files from the non-target and pre-target folders.
    non_target_features = [file for file in os.listdir(non_target_folder) if file.endswith('.png')]
    pre_target_features = [file for file in os.listdir(pre_target_folder) if file.endswith('.png')]

    features = []

    # Convert all non-target images into flattened numpy arrays and add to the features list.
    for path in [os.path.join(non_target_folder, file) for file in non_target_features]:
        img = Image.open(path)
        img_array = np.array(img).flatten()
        features.append(img_array)

    # Convert all pre-target images into flattened numpy arrays and add to the features list.
    for path in [os.path.join(pre_target_folder, file) for file in pre_target_features]:
        img = Image.open(path)
        img_array = np.array(img).flatten()
        features.append(img_array)

    # Create feature and label arrays, where pre-target images are labeled 1 and non-target images are labeled 0.
    X = np.array(features)
    y = np.array([1] * len(pre_target_features) + [0] * len(non_target_features))

    # Split data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

    # Initialize and train a logistic regression model.
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate and print the model's accuracy on the test set.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.1f}%".format(accuracy * 100))

    # Predict the pattern for new images in the last_n_bar_folder.
    new_images = os.listdir(last_n_bar_folder)
    new_image_paths = [os.path.join(last_n_bar_folder, img) for img in new_images]

    print("***PREDICTIONS***")

    # For each new image, predict if it matches the target pattern and print the result.
    for image_path in new_image_paths:
        ticker = image_path.split('\\')[-1].split('_')[0]  # Extracting the ticker from the filename.

        img = Image.open(image_path)
        img_array = np.array(img).flatten().reshape(1, -1)  # Flattening and reshaping the image for prediction.

        prediction = model.predict(img_array)

        # Print the prediction for each ticker.
        if prediction == 1:
            print(f"{ticker}: Target")
        else:
            print(f"{ticker}: Non-Target")


predict_target_images(non_target_folder_name, pre_target_folder_name, last_n_bar_folder_name)
