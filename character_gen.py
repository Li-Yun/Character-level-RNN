import os
import argparse 
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from RNNModel import CharacterRNN

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')


def data_preprocessing(in_data, label_num):
    
    one_hot_encoding = np.zeros((in_data.size, label_num), dtype=np.float32)

    for idx, number in enumerate(in_data.flatten()):
        one_hot_encoding[idx, number] = 1.0

    # reshape the array back to the original arrary
    return one_hot_encoding.reshape((*in_data.shape, label_num))


def load_text_data(file_path):
    output_dic = dict()

    # read file
    with open(file_path, 'r') as fp:
        text_data = fp.read()

    # Tokenization
    # Two dictionaries:
    # (1) int2char: mapping integers to characters
    # (2) char2int: mapping characters to unique integers
    char_tuple = tuple(set(text_data))
    int2char = dict(enumerate(char_tuple))  
    char2int = {val: key for key, val in int2char.items()}
    encoded_text = np.array([char2int[char] for char in text_data])
    output_dic['i2c'] = int2char
    output_dic['c2i'] = char2int
    output_dic['encoded_t'] = encoded_text
    output_dic['char_tup'] = char_tuple

    return output_dic


def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''

    # Get the number of batches we can make
    n_batches = len(arr) // (batch_size * seq_length)

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size * seq_length]

    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))  # number of sequences by (batch number * seq_length)

    # Iterate over the batches using a window of size seq_length
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n + seq_length]

        # The targets, shifted by one
        y = np.zeros(x.shape)
        y[:, :-1] = x[:, 1:]
        try:
            y[:, -1] = arr[:, n+seq_length]
        except IndexError:
            y[:, -1] = arr[:, 0]

        yield x, y

def predict(net, single_char, h=None, top_k=None):
    """
    Given a character, predict the next character.
    Returns the predicted character and the hidden state.
    """
    # tensor inputs
    x = np.array([[net.char2int[single_char]]])
    x = data_preprocessing(x, len(net.chars))
    inputs = torch.from_numpy(x)

    inputs = inputs.cuda() if train_on_gpu else inputs
    
    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)
    
    # get the character probabilities
    p = F.softmax(out, dim=1).data
    p = p.cpu() if train_on_gpu else p
    
    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
        
    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())
    
    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h


def generate_text(net, size, prime='The', top_k=None):

    if train_on_gpu:
        net.cuda()
    else:
        net.cpu()
    
    # eval mode
    net.eval()

    # run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1, train_on_gpu)  # batch size is 1.
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)
    chars.append(char)

    # pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)
    
    return ''.join(chars)


def train(net, data, path, epochs=10, batch_size=10, seq_length=50, 
          lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' 
    Training a network
    '''
    net.train()

    adam_opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    counter = 0
    n_chars = len(net.chars)
    # training
    for epoch in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size, train_on_gpu)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = data_preprocessing(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we would backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            adam_opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size, train_on_gpu)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = data_preprocessing(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we would backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if train_on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())

                    val_losses.append(val_loss.item())

                # reset to train mode after iterationg through validation data
                net.train()

                print("Epoch: {}/{}...".format(epoch+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
    
    # saving the model
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'tokens': net.chars}
    with open(path, 'wb') as f:
        torch.save(checkpoint, f)


def load_model(model_path):
    
    with open(model_path, 'rb') as f:
        checkpoint = torch.load(f)
    
    trained_net = CharacterRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    trained_net.load_state_dict(checkpoint['state_dict'])

    return trained_net


def main():
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file_path', type=str, default='', help='path to a text file')
    parser.add_argument('--batch_size', type=int, default=10, help='number of sequences in mini-batch')
    parser.add_argument('--max_epoch', type=int, default=20, help='maximum epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_num', type=int, default=128, help='number of hidden units in the hidden layer')
    parser.add_argument('--layer_num', type=int, default=2, help='number of layers in LSTM')
    parser.add_argument('--saved_model_path', type=str, default='', help='path to a saved model')
    parser.add_argument('--sequence_length', type=int, default=50, help='sequence length of the input sequence')
    parser.add_argument('--train', type=int, default=1, help='flag to perform model training')
    parser.add_argument('--test', type=int, default=0, help='flag to test a well-trained model')

    args = parser.parse_args()
    text_file_path = os.path.abspath(args.text_file_path)
    model_path = os.path.join(os.path.abspath(args.saved_model_path), 'rnn_model.net')
    b_size = args.batch_size
    max_e = args.max_epoch
    lr_val = args.lr
    n_hidden = args.hidden_num
    n_layers = args.layer_num
    seq_len = args.sequence_length
    training_flag = args.train
    testing_flag = args.test

    clip_val = 5
    val_frac_val = 0.1
    print_every_val = 10

    # training phase
    if training_flag and not testing_flag:
        # load data
        data_obj = load_text_data(text_file_path)
        
        # load a LSTM model
        net = CharacterRNN(data_obj['char_tup'], n_hidden, n_layers)
        if train_on_gpu:
            net.cuda()
        
        # model training
        train(net, data_obj['encoded_t'], model_path, epochs=max_e, batch_size=b_size, 
              seq_length=seq_len, lr=lr_val, clip=clip_val, val_frac=val_frac_val,
              print_every=print_every_val)
    # ===============
    # testing phase
    if not training_flag and testing_flag:
        # load a well-trained model
        trained_char_net = load_model(model_path)
        
        # Sample using a loaded model
        gen_text = generate_text(trained_char_net, 2000, top_k=5, prime="And Levin said")
        
        wfp = open('generated_text.txt', 'wt', encoding='utf-8')
        wfp.write(gen_text)


if __name__=="__main__":
    main()
