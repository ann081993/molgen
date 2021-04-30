#### Model training ----
# Author: Seungchan An

input_shape = as.integer(c(embed - 1, length(ref_char)))
output_dim = as.integer(length(ref_char))
latent_dim = 64
lstm_dim = 64
cat("input_shape, output_dim, latent_dim, lstm_dim:", 
    input_shape, output_dim, latent_dim, lstm_dim, "\n")

unroll = FALSE

