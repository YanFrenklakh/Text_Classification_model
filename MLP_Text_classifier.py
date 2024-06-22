# tqdm
from tqdm import tqdm
# pandas and gpu accelerated pandas
import pandas as pd
import cudf
# NLP
import spacy
import en_core_web_trf
import nltk
# scipy
import scipy
# PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# CSV interaction
import csv
# sklern
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nlp = en_core_web_trf.load()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------  PyTorch Train Dataset  ----------------------------------------------#
class TrainDataset(Dataset):
    STOPWORDS = nltk.corpus.stopwords.words('english')
    filters = [ '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/',  '\\', ':', ';', '<', '=', '>',
                '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '\t','\n',"'",",",'~' , '—']
    
    def __init__(self, df):
        # Make data csv file
        if type(self) is TrainDataset:
            df.to_csv('~/MLP/data/train_data.csv')
        else:
            df.to_csv('~/MLP/data/test_data.csv')
        # Get lists of text data end encoded category 
        # Preform normalization and preproccesing
        self.text = self.preprocess_text_df(df, filters=self.filters)['text'].to_arrow().to_pylist()
        self.category = df['encoded_category'].to_arrow().to_pylist()

    def preprocess_text(self, input_strs , filters=None , stopwords=STOPWORDS):
        """
            * filter punctuation
            * to_lower
            * remove stop words (from nltk corpus)
            * remove multiple spaces with one
            * remove leading spaces    
        """
    
        # filter punctuation and case conversion
        translation_table = {ord(char): ord(' ') for char in filters}
        input_strs = input_strs.str.translate(translation_table)
        input_strs = input_strs.str.lower()
        
        # remove stopwords
        stopwords_gpu = cudf.Series(stopwords)
        input_strs =  input_strs.str.replace_tokens(self.STOPWORDS, ' ')
        
        # replace multiple spaces with single one and strip leading/trailing spaces
        input_strs = input_strs.str.normalize_spaces( )
        input_strs = input_strs.str.strip(' ')
    
        return input_strs
    
    def preprocess_text_df(self, df, text_cols=['text'], **kwargs):
        for col in text_cols:
            df[col] = self.preprocess_text(df[col], **kwargs)
        return  df
    
    def set_text(self, text):
        self.text = torch.tensor(scipy.sparse.csr_matrix.todense(text)).float()

    def get_text(self):
        return self.text

    def __len__(self):
        return len(self.category)

    def __getitem__(self, index):
        return self.text[index], self.category[index]

# --------------------------------------  PyTorch Test Dataset  ----------------------------------------------#
class TestDataset(TrainDataset):
    
    def __getitem__(self, index):
        return self.text[index], self.category[index]

# --------------------------------------  Model  ----------------------------------------------#
class MLP(nn.Module):
    # Multylayer Perceptron
    
    def __init__(self, data, category):
        super().__init__()
        self.layers = nn.Sequential(
                    nn.Linear(data.shape[1], 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, category.nunique()),
                    nn.Dropout(0.2),
                    nn.LogSoftmax(dim=1)
                )
    
    def forward(self, x):
        # Forward pass
        return self.layers(x)

def tdif_transform(x_train, x_test):
    kwargs = {
        'ngram_range': (1,2),  # Use 1-grams + 2-grams.
        'analyzer': 'word',  # Split text into word tokens.
        'min_df': 2,
    }
    vectorizer = TfidfVectorizer(**kwargs)
    # Learn vocabulary from training texts and vectorize training texts.
    x_train_transformed = vectorizer.fit_transform(x_train)
    # Vectorize validation texts.
    x_test_transformed = vectorizer.transform(x_test)
    return x_train_transformed, x_test_transformed

# Train the Modle
def train_model(epochs=10):
    # Set model to train mode
    mlp.train()
    
    # Start epoch loop
    for epoch in range(epochs):
        losses = []
        # Get cutches form dataloder
        for batch_num, input_data in enumerate(train_loader):
            # Zero out the gradieants
            optimizer.zero_grad()
            
            # Get text traning vactor and clasification vector
            x, y = input_data
            # Load data in to GPU
            x = x.to(device)
            y = y.to(device)
            
            # Get resolts from forward pass on the model
            output = mlp(x)

            # Running backward pass wirh output and target
            loss = criterion(output, y)
            # Claculate the gradients 
            loss.backward()
            losses.append(loss.item())

            # Update the internal wheits
            optimizer.step()

            if batch_num % 40 == 0:
                print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))

# Evalueate the Model

def evalueate_model():
    # Set model in to tesing mode
    mlp.eval()

    # Make CSV file to stor the resolts of the text classifications
    with open('~/MLP/data/mlp_resolts.csv', 'w') as f:
        # Hold fileds of text-id and classfication category
        fieldnames = ['text', 'category']
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator = '\n')
        writer.writeheader()
        # Init first text
        text_id = 1
    
        # Set Torch to not update the gradients.
        with torch.no_grad():
            # Get data form the test DataLoader. 
            for _, input_data in enumerate(test_loader):
                # Load the input in to the GPU.
                x, y = input_data
                x = x.to(device).float()
                y = y.to(device)
            
                # Run the text tensor on the model and get results.
                output = mlp(x)
                # Get new tensor with the exponential of the elements of the input tensor.
                ps = torch.exp(output)
                # Calculate test accuresy
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == y.view(*top_class.shape)
                test_accuracy = torch.mean(equals.float())
                print(f"Test Accuracy: {test_accuracy:.3f}")
            
                # Get max value of all elements in the input tensor
                output = output.argmax(dim=1)
                for cls in output:
                    # Write the resolt of the clasification for a given text in the ouput CSV file
                    writer.writerow({fieldnames[0]: text_id,fieldnames[1]: cls.item()})
                    text_id += 1


if __name__ == "__main__":
    # Load the csv data
    df = pd.read_csv('~/MLP/data/bbc-text.csv')

    # Encode data category
    df['encoded_category'] = LabelEncoder().fit_transform(df['category'].to_numpy())

    print('Data frame shape: ', df.shape)
    train, test = train_test_split(df[['text','encoded_category']], test_size=.2, stratify=df['category'], random_state=42)

    # Conver to GPU accelerated pandas
    train_df = cudf.from_pandas(train)
    test_df = cudf.from_pandas(test)

    # Form cuda accelarated dataset
    train_set = TrainDataset(train_df)
    test_set = TestDataset(test_df)

    # Preform tf-IDF vectorizetion
    x_train, x_test = tdif_transform(train_set.get_text(), test_set.get_text())
    train_set.set_text(x_train)
    test_set.set_text(x_test)

    # setup dataLoader for model traning
    train_loader = DataLoader(train_set, batch_size = 64, shuffle=True, num_workers = 4)
    test_loader = DataLoader(test_set, batch_size = 64, shuffle=False, num_workers = 4)


    # Init MLP model
    mlp = MLP(train_set.get_text(),df['category']).to(device)
    print(mlp)

    # Setup Optimizer and Loss function
    optimizer = torch.optim.Adam(mlp.parameters())
    criterion = nn.CrossEntropyLoss()

    # Run model trainin on 10 epochs (insert diferent value to diferent number of epoch)
    train_model()
    
    # Evlauete the modeld and write the resolts in to CSV file
    evalueate_model()
