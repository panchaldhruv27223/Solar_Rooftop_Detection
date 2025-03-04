import re

def get_information(log_file_path):
    '''
        Right now this function is extract information from a log file.
        This function reads the log file and extracts the epoch number and loss value from each line.
        It then return the extracted values as lists.
    '''
    
    # Regular expression to match loss values
    pattern = r"Epoch (\d+) completed\. Loss: ([\d\.]+)"

    # Lists to store extracted values
    epochs, losses = [], []

    # Read and parse the log file
    with open(log_file_path, "r") as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)

    # Print extracted values
    # print("Epochs:", epochs)
    # print("Losses:", losses)
    return epochs, losses


if __name__ == "__main__":
    
    # log_file_path = "/home/selc-a4-sr2/Solar_Rooftop_Detection/logs/log_2025-02-25_16-49-25.log"
    log_file_path = "/home/selc-a4-sr2/Solar_Rooftop_Detection/logs/log_2025-02-22_09-23-19.log"
    print(get_information(log_file_path))
    print("Dhruv")