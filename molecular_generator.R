#### 2020-09-11 molecular_generator.R
# Author: Seungchan An

# Ref**: https://www.cheminformania.com/master-your-molecule-generator-seq2seq-rnn-models-with-smiles-in-keras/
# Ref: https://buomsoo-kim.github.io/keras/2019/07/12/Easy-deep-learning-with-Keras-19.md/
# Ref: release
source("https://raw.githubusercontent.com/ann081993/fp_util/master/fp_util_src.R")
library(keras)

#setwd("/home/oem/ml/release")
#aa <- read.csv("chembl_mw_ro5_filtered_200911.csv", stringsAsFactors = FALSE, header = FALSE)
#write.table(aa, "chembl_mw_ro5_filtered_200911.smi", sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
#smi <- read.csv("chembl_mw_ro5_filtered_200911.smi", stringsAsFactors = FALSE, header = FALSE, sep = "\t")[, 1]

#### Load ChEMBL
aa <- read.csv("chembl_mw150-450_ro5_filtered_200911.csv.gz", stringsAsFactors = FALSE, header = TRUE, sep = ";")
smiles <- aa$Smiles

summary(nchar(smiles))
smiles <- smiles[nchar(smiles) < 100 & nchar(smiles) > 5]

#smi <- smiles[50001:200000]

#smi <- c(smiles[1:10], "ADSFSADFDFDFDF")
#a <- parse.smiles(smiles[1:10000])
#b <- sapply(a, is.null)
#!is.null(parse.smiles(smi)[[1]])

#### Load PubChem
#aa <- readLines("/mnt/sdd1/pubchem/CID-SMILES.gz") # 109909664
#aa <- gsub("^[0-9]*\t", "", aa)
#summary(nchar(aa))
#aa <- aa[nchar(aa) < 300] # 109763835
#aa <- aa[!grepl("[.]", aa)] # 103409114
#writeLines(aa, "PubChem SMILES.txt")
smiles <- readLines("PubChem SMILES.txt", n = 1000000)
smiles <- smiles[!grepl("D|E|G|R|T|U|V|W|X|Y", smiles)]

#aa[!grepl("CC|C1|C\\(", aa)]


#head(smiles[grep("E", smiles)])

# tokenization
ref_char <- unique(unlist(strsplit(smiles, split = "")))
ref_char <- c("!", ref_char[order(ref_char)], "E") # E stands for empty
embed <- max(nchar(smiles)) + 5
length(ref_char); embed

smi2one_hot <- function(smi_input) {
        smi_input <- paste0("!", smi_input)
    
        if(length(smi_input) > 1) {
                one_hot <- array(0, dim = c(length(smi_input), embed, length(ref_char)))
                
                for(sn in 1:length(smi_input)) {
                        for(n in 1:embed) {
                                one_hot[sn, n, which(ref_char == substr(smi_input[sn], n, n))] <- 1
                        }
                        one_hot[sn, (nchar(smi_input[sn]) + 1):embed, which(ref_char == "E")] <- 1
                }
                cat("\n", length(smi_input), "SMILES queried\n",
                    "Dimension of output array:", dim(one_hot), "\n")
                return(one_hot)
        } else {
                one_hot <- matrix(0, nrow = embed, ncol = length(ref_char))
                for(n in 1:embed) {
                        one_hot[n, which(ref_char == substr(smi_input, n, n))] <- 1
                }
                one_hot[(nchar(smi_input) + 1):embed, which(ref_char == "E")] <- 1
                cat("\n", length(smi_input), "SMILES queried\n",
                    "Dimension of output matrix:", dim(one_hot), "\n")
                return(one_hot)
        }
}

one_hot2smi <- function(one_hot) {
        smi_char <- NULL
        for(n in 1:embed) {
                smi_char <- c(smi_char, ref_char[which(one_hot[n, ] == 1)])
        }
        return(paste0(smi_char, collapse = ""))
}


#array(1:24, c(3, 4, 2))[-1,,]

smi_array <- smi2one_hot(smi)
dim(smi_array)
xtrain <- smi_array[,-dim(smi_array)[2],]
ytrain <- smi_array[,-1,]

#smi[2]
#one_hot2smi(smi_array[2,,])

# model
library(keras)
#input_shape = dim(xtrain)[2:3]
#output_dim = dim(ytrain)[3]
input_shape = as.integer(c(embed - 1, length(ref_char)))
output_dim = as.integer(length(ref_char))
latent_dim = 64
lstm_dim = 64
cat("input_shape, output_dim, latent_dim, lstm_dim:", 
    input_shape, output_dim, latent_dim, lstm_dim, "\n")

unroll = FALSE

#### Seq2seq ----
# encoder
encoder_inputs <- layer_input(shape = input_shape)
encoder <- layer_lstm(units = lstm_dim,
                      return_state = TRUE,
                      unroll = unroll)
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
decoder_lstm <- layer_lstm(units = lstm_dim,
                           return_sequences = TRUE,
                           unroll = unroll)
decoder_outputs <- decoder_lstm(decoder_inputs, initial_state = encoder_states)
decoder_dense <- layer_dense(units = output_dim, activation = 'softmax')
decoder_outputs <- decoder_dense(decoder_outputs)

# define model
model <- keras_model(c(encoder_inputs, decoder_inputs), decoder_outputs)
model

model %>% compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer_adam(lr=0.005),
                  metrics = c('accuracy'))

history <- model %>% fit(
        list(xtrain, xtrain), ytrain, 
        epochs = 1000, batch_size = 256, 
        validation_split = 0.2,
        shuffle = TRUE
)

append_fit <- function(f1, f2) {
    f1$params$epochs <- f1$params$epochs + f2$params$epochs
    f1$params$samples <- f1$params$samples + f2$params$samples
    for(n in names(f1$metrics)) {
        f1$metrics[[n]] <- c(f1$metrics[[n]], f2$metrics[[n]])
    }
    return(f1)
}

# train batches
set.seed(5582)
smiles <- sample(smiles, length(smiles))

size_per_batch = 20000
parts <- ceiling(length(smiles) / size_per_batch)

history <- NULL
for(pt in 1:parts) {
    from = (size_per_batch * (pt - 1) + 1)
    to = (size_per_batch * (pt - 1) + ifelse((parts == pt) & length(smiles) %% size_per_batch > 0, length(smiles) %% size_per_batch, size_per_batch))
    
    smi <- smiles[from:to]
    smi_array <- smi2one_hot(smi)
    xtrain <- smi_array[,-dim(smi_array)[2],]
    ytrain <- smi_array[,-1,]
    
    h <- model %>% fit(
        list(xtrain, xtrain), ytrain, 
        epochs = 10, batch_size = 256, 
        validation_split = 0.2,
        shuffle = TRUE
    )
    cat("... Training SMILES", from, "-", to, ":", pt, "of", parts, "\n")
    
    if(is.null(history)) {
        history <- h 
    } else {
        history <- append_fit(history, h)
    }
}

save_model_hdf5(object = model, filepath = "./seq2seq_all_10K_per_epoch_20.h5", include_optimizer = TRUE)
model <- load_model_hdf5("./seq2seq_all_10K_per_epoch_20.h5")


save_model_hdf5(model, filepath = "Model generator seq2seq v0 molecules 10K epoch 500 200913")
pdf("Model generator seq2seq v0 molecules 10K epoch 500 200913.pdf", 5, 5)
plot(history)
dev.off()

# smiles to latent model
smiles2latent_model <- keras_model(inputs = encoder_inputs,
                                   outputs = neck_outputs)

latent_input =  layer_input(shape = latent_dim)
state_h_decoded_2 =  decode_h(latent_input)
state_c_decoded_2 =  decode_c(latent_input)
latent2states_model <- keras_model(inputs = latent_input,
                                   outputs = c(state_h_decoded_2, state_c_decoded_2))

inf_decoder_inputs = layer_input(batch_shape=c(1, 1, input_shape[2]))
inf_decoder_lstm = layer_lstm(units = lstm_dim,
                              return_sequences = TRUE,
                              unroll = unroll,
                              stateful = TRUE) #**
inf_decoder_outputs = inf_decoder_lstm(inf_decoder_inputs)
inf_decoder_dense = layer_dense(units = output_dim, activation='softmax')
inf_decoder_outputs = inf_decoder_dense(inf_decoder_outputs)
sample_model = keras_model(inf_decoder_inputs, inf_decoder_outputs)

for(i in 2:3) {
        set_weights(object = sample_model$layers[[i]],
                    weights = get_weights(object = model$layers[[i + 6]]))
}

# fingerprint
x_latent = smiles2latent_model %>% predict(xtrain)

molno = 20
latent_mol = smiles2latent_model %>% predict(xtrain[molno:(molno+1),,])

result <- NULL
for(n in 1:nrow(x_latent)) {
        result <- c(result, 
                    dist(rbind(x_latent[n, ], latent_mol[1, ])))
}
which(rank(result) < 10)
smi[which(rank(result) < 10)]
smi[molno]

# latent to smiles model
get_latent <- function(smi) {
    smi_array <- smi2one_hot(c(smi, smi))
    
    input <- smi_array[,-dim(smi_array)[2],]
    latent = smiles2latent_model %>% predict(input)
    return(latent[1,,drop=FALSE])
}

latent2smiles_model <- function(latent) {
        #Decode states and set Reset the LSTM cells with them
        states <- latent2states_model %>% predict(latent)
        
        k_set_value(sample_model$layers[[2]]$states[[1]], value = states[[1]])
        k_set_value(sample_model$layers[[2]]$states[[2]], value = states[[2]])
        
        #sample_model$layers[[2]]$states
        #,states = c(states[[1]], states[[2]])
        #sample_model.layers[1].reset_states(states=[states[0],states[1]])
        
        #Prepare the input char
        startidx <- which(ref_char == "!")
        samplevec <- array(0, dim = c(1, 1, length(ref_char)))
        samplevec[1,1,startidx] <- 1
        
        smiles = NULL
        #Loop and predict next char
        for(i in 1:embed) {
                o <- sample_model %>% predict(samplevec, batch_size = 1)
                sampleidx <- which.max(o)
                samplechar <- ref_char[sampleidx]
                #cat(samplechar)
                if(samplechar != "E") {
                        smiles = c(smiles, samplechar)
                        samplevec <- array(0, dim = c(1, 1, length(ref_char)))
                        samplevec[1,1,sampleidx] <- 1
                } else {
                        break
                }
        }
        return(paste(smiles, collapse = ""))
}
latent2smiles_model(get_latent(smi[n]))

smi[1]
latent2smiles_model(x_latent[1,, drop = FALSE])


latent2smiles_model(get_latent("CCCC"))

latent2smiles_model(get_latent(smi[n])); smi[n]; n=n+1

n = 1
a = latent2smiles_model(get_latent(smi[n])); b = gsub("!","",smi[n]); n=n+1; a; b; plot_smi(c(a,b), nrow=1,ncol=2)


ex_smi <- "CCOC(=O)C(Cc1ccccc1)C(=O)NCc1ccccc1"
one_hot2smi(ex_smi)
get_latent(ex_smi)
ex_latent <- get_latent(ex_smi)
latent2smiles_model(ex_latent)
a = latent2smiles_model(get_latent(ex_smi)); b = ex_smi; a; b; plot_smi(c(a,b), nrow=1,ncol=2)


ex_smi <- "CCN(c1ccccc1)S(=O)(=O)c1ccc(C(=O)NCCCN2CCOCC2)cc1"

ex_smi <- "O=C(O)CCC(=O)Nc1ccc(COc2ccccc2F)cc1"



mol_gen <- function(query_smi = "O=C(O)c1ccccc1",
                    dev = 0.1, n = 5, time_out = 10) {
    t0 = Sys.time()
    latent <- get_latent(query_smi)
    output_smi <- NULL
    while(length(output_smi) < n & as.numeric(Sys.time() - t0) < time_out) {
        s <- latent2smiles_model(latent + array(rnorm(64, sd = dev), dim = c(1,64)))
        if(!is.null(parse.smiles(s)[[1]])) {
            output_smi <- unique(c(output_smi, s))
        }
    }
    return(output_smi)
}
output_smi <- mol_gen(dev = 0.1)
plot_smi(c(query_smi, output_smi), nrow = 2, ncol = 3)

output_smi <- mol_gen(ex_smi, dev = 0.001)
plot_smi(c(ex_smi, output_smi), nrow = 2, ncol = 3)




#### Seq2seq + Attention ----
# Ref: https://neurowhai.tistory.com/291
input_shape = as.integer(c(embed - 1, length(ref_char)))
output_dim = as.integer(length(ref_char))
latent_dim = 64
lstm_dim = 64
cat("input_shape, output_dim, latent_dim, lstm_dim:", 
    input_shape, output_dim, latent_dim, lstm_dim, "\n")

# encoder
encoder_inputs <- layer_input(shape = input_shape)
encoder <- layer_gru(units = lstm_dim,
                     return_state = TRUE, return_sequences = TRUE)
encoder_outputs <- encoder(encoder_inputs)
state_h <- encoder(encoder_inputs)

# decoder
decoder_inputs <- layer_input(shape = input_shape)
decoder <- layer_gru(units = lstm_dim,
                     return_state = TRUE, return_sequences = TRUE)
decoder_outputs <- decoder(decoder_inputs, initial_state = state_h)

# attention
flatten_h <- layer_reshape(encoder_outputs, target_shape = input_shape[1] * lstm_dim)
repeat_h <- layer_repeat_vector(flatten_h, n = input_shape[1])
repeat_h <- layer_reshape(repeat_h, target_shape = c(input_shape[1] * input_shape[1], lstm_dim))

repeat_d <- layer_lambda(

layer_repeat_vector
flatten_h = layers.Reshape((max_encoder_seq_length * latent_dim,))(encoder_outputs) repeat_h = layers.RepeatVector(max_decoder_seq_length)(flatten_h) repeat_h = layers.Reshape((max_encoder_seq_length * max_decoder_seq_length, latent_dim))(repeat_h) repeat_d = layers.Lambda(lambda x: K.concatenate([K.repeat(x[:, i, :], max_encoder_seq_length) for i in range(0, max_decoder_seq_length)], axis=-2), lambda x: tuple([x[0], max_encoder_seq_length * max_decoder_seq_length, latent_dim])) repeat_d = repeat_d(decoder_outputs)




decode_h <- layer_dense(units = lstm_dim, activation = 'relu')
decode_c <- layer_dense(units = lstm_dim, activation = 'relu')
state_h_decoded <- decode_h(neck_outputs)
state_c_decoded <- decode_c(neck_outputs)
encoder_states <- c(state_h_decoded, state_c_decoded)
decoder_outputs <- decoder_lstm(decoder_inputs, initial_state = encoder_states)
decoder_dense <- layer_dense(units = output_dim, activation = 'softmax')
decoder_outputs <- decoder_dense(decoder_outputs)

# define model
model <- keras_model(c(encoder_inputs, decoder_inputs), decoder_outputs)
model

model %>% compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer_adam(lr=0.005),
                  metrics = c('accuracy'))

history <- model %>% fit(
    list(xtrain, xtrain), ytrain, 
    epochs = 1000, batch_size = 256, 
    validation_split = 0.2,
    shuffle = TRUE
)

append_fit <- function(f1, f2) {
    f1$params$epochs <- f1$params$epochs + f2$params$epochs
    f1$params$samples <- f1$params$samples + f2$params$samples
    for(n in names(f1$metrics)) {
        f1$metrics[[n]] <- c(f1$metrics[[n]], f2$metrics[[n]])
    }
    return(f1)
}

