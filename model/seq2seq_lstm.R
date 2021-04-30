#### Seq2seq LSTM ----
# Author: Seungchan An
library(keras)

# encoder
encoder_inputs <- layer_input(shape = input_shape)
encoder <- layer_lstm(units = lstm_dim,
                      return_state = TRUE,
                      unroll = unroll)
encoder_outputs <- encoder(encoder_inputs)[[1]]
state_h <- encoder(encoder_inputs)[[2]]
state_c <- encoder(encoder_inputs)[[3]]
states <- layer_concatenate(inputs = c(state_h, state_c), axis = -1)
neck <- layer_dense(units = latent_dim, activation = 'relu')
neck_outputs <- neck(states)        

# decoder
decode_h <- layer_dense(units = lstm_dim, activation = 'relu')
decode_c <- layer_dense(units = lstm_dim, activation = 'relu')
state_h_decoded <- decode_h(neck_outputs)
state_c_decoded <- decode_c(neck_outputs)
encoder_states <- c(state_h_decoded, state_c_decoded)
decoder_inputs <- layer_input(shape = input_shape)
decoder <- layer_lstm(units = lstm_dim,
                      return_sequences = TRUE,
                      unroll = unroll)
decoder_outputs <- decoder(decoder_inputs, initial_state = encoder_states)
decoder_dense <- layer_dense(units = output_dim, activation = 'softmax')
decoder_outputs <- decoder_dense(decoder_outputs)

# define model
model <- keras_model(c(encoder_inputs, decoder_inputs), decoder_outputs)
print(model)
