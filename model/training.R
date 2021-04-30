#### Model training ----
# Author: Seungchan An

input_shape = as.integer(c(embed - 1, length(ref_char)))
output_dim = as.integer(length(ref_char))
latent_dim = 64
lstm_dim = 64
cat("input_shape, output_dim, latent_dim, lstm_dim:", 
    input_shape, output_dim, latent_dim, lstm_dim, "\n")

unroll = FALSE


source("https://raw.githubusercontent.com/ann081993/molgen/main/model/seq2seq_lstm.R")
source("https://raw.githubusercontent.com/ann081993/molgen/main/model/seq2seq_gru.R")
