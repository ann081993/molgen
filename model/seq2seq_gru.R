#### Seq2seq GRU ----
# Author: Seungchan An
library(keras)

# encoder
encoder_inputs <- layer_input(shape = input_shape)
encoder <- layer_gru(units = lstm_dim, return_state = TRUE) # 
encoder_outputs <- encoder(encoder_inputs)[[1]]
state_h <- encoder(encoder_inputs)[[2]]

# decoder
decoder_inputs <- layer_input(shape = input_shape)
decoder <- layer_gru(units = latent_dim, return_sequences = TRUE) # 
decoder_outputs <- decoder(decoder_inputs, initial_state = state_h)
decoder_dense = layer_dense(units = output_dim, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model <- keras_model(list(encoder_inputs, decoder_inputs), decoder_outputs)
model

# define model
model <- keras_model(c(encoder_inputs, decoder_inputs), decoder_outputs)
print(model)
