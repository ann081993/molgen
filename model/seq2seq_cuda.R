#### Seq2seq CUDA ----
# Author: Seungchan An

# encoder
encoder_inputs <- layer_input(shape = input_shape)
encoder <- layer_cudnn_lstm(units = lstm_dim, return_state = TRUE)
encoder_outputs <- encoder(encoder_inputs)
state_h <- encoder(encoder_inputs)
state_c <- encoder(encoder_inputs)
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
decoder_lstm <- layer_lstm(units = lstm_dim, return_sequences = TRUE)
decoder_outputs <- decoder_lstm(decoder_inputs, initial_state = encoder_states)
decoder_dense <- layer_dense(units = output_dim, activation = 'softmax')
decoder_outputs <- decoder_dense(decoder_outputs)

# define model
model <- keras_model(c(encoder_inputs, decoder_inputs), decoder_outputs)
model
