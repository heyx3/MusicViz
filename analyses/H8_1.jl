using Random, Statistics, LinearAlgebra,
      Flux, ProgressMeter, PyPlot,
      MusicViz
(() -> begin

aud = grab_mono_from_wav("C:/Users/manni/Desktop/H8_1_Audio.wav")
chunks = nn_chunked_audio(aud..., Int(round(aud[2])), Int(round(aud[2]/2)))

input_settings = chunks.input_settings
chunk = first(chunks)
settings = NN_Settings(input_settings=input_settings)

input_size_2D = nn_input_size(input_settings, chunks.chunk_size_samples)
input_size_3D = nn_chunked_input_size(chunks)

chain_en = nn_chain_encoder(settings)
chain_dec = nn_chain_decoder(settings)
display(chain_en)
display(chain_dec)
chain_full = Chain(chain_en, chain_dec)

x0 = rand(Float32, input_size_2D..., 1, 1)
L = chain_en(x0)
x1 = chain_dec(L)
display(size(x0))
display(size(x1))
display(Flux.Losses.mse(x1, x0))

PROPORTION_TRAINING = 0.15
PROPORTION_VERIFICATION = 0.05
#NOTE: the last batch is padded with zeros so possibly a bad training example; skip it in the training batches.
training_size_3D = (
    input_size_3D[1:2]...,
    max(1, Int(round((input_size_3D[3]-1) * PROPORTION_TRAINING)))
)
verification_size_3D = (
    input_size_3D[1:2]...,
    clamp(Int(round(input_size_3D[3] * PROPORTION_VERIFICATION)),
          0, input_size_3D[3] - training_size_3D[3])
)
println("Training on ", training_size_3D[3], ", then verifying on ", verification_size_3D[3])

train_batches = fill(0.0f0, training_size_3D[1:2]..., 1, training_size_3D[3])
verify_batches = fill(0.0f0, verification_size_3D[1:2]..., 1, verification_size_3D[3])
full_batches = fill(0.0f0, input_size_3D[1:2]..., 1, input_size_3D[3])
for (i,b) in enumerate(chunks)
    full_batches[:, :, 1, i] = b
end
training_batch_idcs = Int[ ]
used_batches = Set{Int}()
for i in 1:training_size_3D[3]
    j = clamp(Int(round(input_size_3D[3] * i / training_size_3D[3])),
              1, input_size_3D[3]-1)
    if in(j, used_batches)
        for k in (maximum(used_batches)+1):(input_size_3D[3]-1)
            if !in(k, used_batches)
                j = k
                break
            end
        end
    end
    if !in(j, used_batches) # Possibly we run out near the end of the batch list, due to integer math
        push!(training_batch_idcs, j)
        push!(used_batches, j)
        train_batches[:, :, 1, i] = full_batches[:, :, 1, j]
    end
end
verify_batch_idcs = Int[ ]
prev_j::Int = 1
for i in 1:verification_size_3D[3]
    j = clamp(Int(round(input_size_3D[3] * i / verification_size_3D[3])),
              1, input_size_3D[3])
    if in(j, used_batches)
        for k in prev_j:input_size_3D[3]
            if !in(k, used_batches)
                j = k
                break
            end
        end
    end
    prev_j = j
    if !in(j, used_batches) # Possibly we run out near the end of the batch list, due to integer math
        push!(verify_batch_idcs, j)
        push!(used_batches, j)
        verify_batches[:, :, 1, i] = full_batches[:, :, 1, j]
    end
end
println("Arbitrary sample from the full batch: ", full_batches[3, 3, 1, 3])

grouped_batches = Flux.DataLoader(train_batches, batchsize=128, shuffle=true, rng=Random.Xoshiro(0x5555ffaf))
loaded_grouped_batches::Vector{Array{Float32, 4}} = collect(grouped_batches)
println("Individual training batches:", (b -> (typeof(b), size(b))).(loaded_grouped_batches))

loss_function_factory(batch) = model -> begin
    (encoder, decoder) = model.layers
    result_encoded = encoder(batch)
    result_decoded = decoder(result_encoded)

    loss_decoding = Flux.Losses.mse(result_decoded, batch)

    # Covariance is normally computed with 'cov()', but that mutates internally
    #    so it breaks the gradient calculation :(
    encoding_covariance = let z = Matrix(reshape(result_encoded, size(result_encoded, 1), :))'
        μ = mean(z, dims=2)
        zc = z .- μ
        (zc * zc') / (size(z, 2) - 1)
    end
    loss_correlation = if true
        v = diag(encoding_covariance) .+ Ref(0.0000001f0)
        D = Diagonal(1 ./ sqrt.(v))
        R = D * encoding_covariance * D

        d = size(R, 1)
        R_off_diagonal = R .- Matrix{eltype(R)}(I, d, d)
        sum(R_off_diagonal .^ 2)
    else
        correlation = encoding_covariance .- Diagonal(diag(encoding_covariance))
        loss_correlation = sum(correlation .^ 2)
    end

    return (1.0f0 * loss_decoding) + (0.001f0 * loss_correlation)
end

trainer = Flux.setup(Adam(), chain_full)
@showprogress "Training..." for epoch in 1:100
    for batch in grouped_batches
        gradients  = Flux.gradient(loss_function_factory(batch), chain_full)
        Flux.update!(trainer, chain_full, gradients[1])
    end
end

println("Baseline Zero-MSE (i.e. MSE for a model that always outputs 0's) against first verification dataset:\n",
        "\t(NOTE: actual model loss function has other terms alongside MSE)\n\t",
        Flux.Losses.mse(verify_batches[:, :, 1:1, 1:1],
                        fill(0.0f0, size(verify_batches)[1:2]..., 1, 1)))
println("Baseline Mean-MSE (i.e. MSE for a model that always outputs mean of inputs) against first verification dataset:\n",
        "\t(NOTE: actual model loss function has other terms alongside MSE)\n\t",
        Flux.Losses.mse(verify_batches[:, :, 1:1, 1:1],
                        fill(mean(verify_batches[:, :, 1, 1]), size(verify_batches)[1:2]..., 1, 1)))

println("\nResults in training batches: [")
mse_training = [ ]
for full_i in training_batch_idcs
    batch = full_batches[:, :, 1:1, full_i:full_i]
    push!(mse_training, loss_function_factory(batch[:, :, 1:1, 1:1])(chain_full))
    println("\t", full_i, " => ", mse_training[end])
end
println("]")
println("μ=", mean(mse_training), "σ=", std(mse_training))
println("\nResults in verification batches: [")
mse_verification = [ ]
for full_i in verify_batch_idcs
    batch = full_batches[:, :, 1:1, full_i:full_i]
    push!(mse_verification, loss_function_factory(batch[:, :, 1:1, 1:1])(chain_full))
    println("\t", full_i, " => ", mse_verification[end])
end
println("]")
println("μ=", mean(mse_verification), "σ=", std(mse_verification),
        "\n")

points_by_node = Vector{Float32}[ ]
for batch_i in 1:input_size_3D[3]
    raw_output = chain_en(full_batches[:, :, 1:1, batch_i:batch_i])
    (batch_i == 1) && println("Encoder output size: ", size(raw_output))
    output = reshape(raw_output, settings.n_outputs)

    for node_i in 1:settings.n_outputs
        (length(points_by_node) < node_i) && push!(points_by_node, Vector{Float32}())
        push!(points_by_node[node_i],
              output[node_i])
    end
end
for node_i in 1:settings.n_outputs
    plot(points_by_node[node_i], label="N$node_i")
end

for (name, plotted_batches_src) in (("Training", train_batches),
                                    ("Verification", verify_batches))
    (py_frame, py_plots) = plt.subplots(1, 2)
    py_plots[1].annotate("$name Original", xy=(0, 0), verticalalignment="bottom")
    py_plots[2].annotate("$name Reconstructed", xy=(0, 0), verticalalignment="bottom")
    py_plots[1].imshow(
        plotted_batches_src[:, :, 1, end],
        cmap="hot", interpolation="nearest"
    )
    py_plots[2].imshow(
        chain_full(plotted_batches_src[:, :, 1:1, end:end])[:, :, 1, 1],
        cmap="hot", interpolation="nearest"
    )
end

end)()