# Implementation of a neural-net approach to visualization.


#################################################
##   Input spectrogram generation

"""
Rules for how to generate the neural network input from an audio sample.

The sample is broken into small overlapping windows, and a running FFT is taken for each window,
    producing a spectrogram in perceptually-linear space.
"""
@kwdef struct NN_InputSettings{TWindow<:Base.Callable}
    window::TWindow = hanning
    window_size_samples::Int = 2048
    window_overlap_samples::Int = window_size_samples ÷ 2
    n_output_frequency_buckets::Int = 64
end

"The size of the neural network's input, produced according to the given settings and with the given input size"
nn_input_size(settings::NN_InputSettings, sample_count::Int)::NTuple{2, Int} = (
    # Audio is a real-valued signal, and analysis doesn't need the Phase,
    #    so only take the magnitudes of the front-half of the complex FFT.
    settings.n_output_frequency_buckets,
    # Turning the window parameters into a count involves a lot of integer math;
    #    DSP.jl already does it for us but there's no easy way to access it :(
    # So this is a copy of the 'k' field computed in `DSP.ArraySplit(...)`.
    (sample_count >= settings.window_size_samples) ?
        ((sample_count - settings.window_size_samples) ÷
         (settings.window_size_samples - settings.window_overlap_samples)
        )+1 :
        0
)

"
Turns an audio sample into a matrix of continuous Float32 FFT results
    by moving a window across the samples.

Returns a matrix where each column is the FFT of one window,
   `settings.n_output_frequency_buckets × window_count`.
You can compute this size with `nn_input_size()`.

The output samples are not in Hz with amplitudes,
   but perceptually-linear Mels with perceptually-linear volume.
"
function nn_get_input_spectrogram(samples::AbstractVector{<:AbstractFloat},
                                  sample_rate::Float32,
                                  settings::NN_InputSettings
                                  ;
                                  buffer::Vector{Float32} = Float32[ ]
                                 )::Matrix{Float32}
    # Normalize the audio samples.
    # This prevents loud sections from being more important than soft ones,
    #    and prevents overfitting by landmarking the loud moments.
    μ = mean(samples)
    σ = std(samples)
    resize!(buffer, length(samples))
    buffer .= (samples .- μ) ./ (σ + @f32(1e-6))

    # Take a continuous sliding-window FFT of the samples.
    stft_samples_full = DSP.stft(buffer, settings.window_size_samples, settings.window_overlap_samples,
                                 window=settings.window, fs=sample_rate)
    stft_samples_mag = abs.(stft_samples_full)

    # Convert FFT frequencies from Hz (physically-linear) to Mels (perceptually-linear).
    (n_stft_bins, n_frames) = size(stft_samples_mag)
    # stft_frequencies = range(start=0.0f0,
    #                          stop=sample_rate/2.0f0,
    #                          length=Float32(n_stft_bins))
    mel_samples = transpose(melscale_filterbanks(n_freqs = n_stft_bins,
                                                 n_mels = settings.n_output_frequency_buckets,
                                                 sample_rate = Int(sample_rate),
                                                 fmin=0.0f0,
                                                 fmax=sample_rate/2)
                           ) * stft_samples_mag
    mel_samples .= log.(mel_samples .+ @f32(1e-6))

    return mel_samples
end

export NN_InputSettings, nn_input_size, nn_get_input_spectrogram



################################################################
##   Splitting an audio file into multiple spectrogram batches

"
A lazy-iterator over the neural network inputs for a larger audio sample.
The samples are split into 'chunks' with a certain duration and stride;
   usually the stride is smaller than the duration so that the chunks overlap.

All inputs will be a matrix of the size `nn_input_size(input_settings, chunk_size_samples)`.

Iteration re-uses allocations internally, so you shouldn't hold onto data from one chunk
   while moving to the next!
You can further eliminate heap allocations by providing `allocation_buffer`,
   but then you shouldn't use this iterator more than once at a time.
"
@kwdef struct nn_chunked_audio{TSamples<:AbstractVector{<:AbstractFloat}, TWindow}
    samples::TSamples
    sample_rate::Float32

    chunk_size_samples::Int = 44100
    chunk_shift_samples = chunk_size_samples ÷ 2

    # Determines how to compute the sliding-window FFT within each chunk.
    input_settings::NN_InputSettings{TWindow} = NN_InputSettings()

    # If provided, prevents more heap allocations but also prevents
    #    using the same iterator more than once at a time.
    allocation_buffer::Optional{Vector{Float32}} = nothing
end
nn_chunked_audio(samples, sample_rate, chunk_size_samples, chunk_shift_samples; kw...) =
    nn_chunked_audio(; samples=samples, sample_rate=sample_rate,
                       chunk_size_samples=chunk_size_samples,
                       chunk_shift_samples=chunk_shift_samples,
                       kw...)

Base.length(chunker::nn_chunked_audio) = (length(chunker.samples) + chunker.chunk_shift_samples - 1) ÷ chunker.chunk_shift_samples

"
Gets the size of the 3D array containing all neural network inputs from the given chunked audio.
In keeping with Flux.jl's standard, the last axis is batch index
   (i.e. each XY slice is a single chunk's spectrogram).
"
nn_chunked_input_size(chunker::nn_chunked_audio)::NTuple{3, Int} = (
    nn_input_size(chunker.input_settings, chunker.chunk_size_samples)...,
    length(chunker)
)

Base.IteratorSize(::Type{<:nn_chunked_audio}) = Base.HasLength()
Base.eltype(::Type{<:nn_chunked_audio}) = Matrix{Float32}

function Base.iterate(chunker::nn_chunked_audio)
    if isempty(chunker.samples)
        return nothing
    end

    return impl_chunked_audio(
        chunker,
        1:chunker.chunk_size_samples,
        exists(chunker.allocation_buffer) ? chunker.allocation_buffer : Float32[ ]
    )
end
function Base.iterate(chunker::nn_chunked_audio, (sample_start, buffers...))
    sample_idcs = sample_start:(sample_start + chunker.chunk_size_samples)
    next_sample_start = sample_start + chunk.chunk_shift_samples
    next_sample_idcs = range(start=next_sample_start,
                             length=length(sample_idcs))

    if first(next_sample_idcs) > length(chunker.samples)
        return nothing
    else
        return impl_chunked_audio(chunker, next_sample_idcs, buffers...)
    end
end

function impl_chunked_audio(chunker::nn_chunked_audio,
                            sample_range::UnitRange{<:Integer},
                            buffer1::Vector{Float32})
    pseudo_spectrogram = nn_get_input_spectrogram(
        samples_with_padding(chunker.samples, sample_range),
        chunker.sample_rate,
        chunker.input_settings,
        buffer = buffer1
    )
    return (pseudo_spectrogram, (first(sample_range), buffer1))
end


function test_chunked_audio(chunker::nn_chunked_audio,
                            make_plots::Bool = true,
                            plots_use_full_spectrograrm::Bool = false)
    estimated_progress_weight = 1 / length(chunker)
    for (i, spectrogram::Matrix{Float32}) in enumerate(chunker)
        if make_plots
            (data, label) = if plots_use_full_spectrograrm
                (spectrogram, "spectrogram $i")
            else
                (spectrogram[:, size(spectrogram, 2)÷2],
                 "periodogram $i")
            end
            plot(data, label=label)
        end
        println("Progress: ", Int(round(100 * min(1.0, i * estimated_progress_weight))), "%")
    end
end

export nn_chunked_audio, nn_chunked_input_size, test_chunked_audio



###################################################
##   Network definition

@kwdef struct NN_Settings{TWindow}
    input_settings::NN_InputSettings{TWindow} = NN_InputSettings()
    # The number of output nodes that will drive the visualizer.
    # Note that in training, this is actually the middle layer --
    #    the first half of the network computes these nodes,
    #    and the second half tries to reconstruct the input from them.
    n_outputs::Int = 8

    # The music will be broken up into chunks of a given size,
    #    usually with overlap between each chunk (by making 'shift' less than 'size').
    chunk_size_samples::Int = 44100
    chunk_shift_samples = chunk_size_samples ÷ 2

    #TODO: Meaningful high-level parameters for convolutional layers
end


"The number of different convolution kernels in the final convolution layer"
const NN_CONVOLUTION_OUT_N_CHANNELS = 64

"Shorthand for the shape of the neural network data after the first convolutional layers"
nn_chain_shape_after_convolution(settings::NN_Settings)::NTuple{3, Int} = (
    ((nn_input_size(settings.input_settings, settings.chunk_size_samples) .+ Ref(3)) .÷ Ref(4))...,
    NN_CONVOLUTION_OUT_N_CHANNELS
)

"
Given a sequence of `Conv` layers eventually followed by corresponding `ConvTranspose` inverse layers,
  computes how much padding to put in each `ConvTranspose` inverse layer.
"
function nn_convolution_layer_padding(input_size::NTuple{2, Int}, n_layers::Int,
                                      stride::NTuple{2, Int},
                                      pad::NTuple{2, Int},
                                      kernel::NTuple{2, Int},
                                      dilation::NTuple{2, Int}
                                     )::Vector{NTuple{2, Int}}
    # Forward conv formula: out = floor((in + 2*pad - dilation*(kernel-1) -1)/stride)+1
    sizes = [ input_size ]
    for i in 1:n_layers
        push!(sizes,
            fld.(
                sizes[end] .+ (Ref(2) .* pad) .- (dilation .* (kernel .- Ref(1))) .- Ref(1),
                stride
            ) .+ Ref(1)
        )
    end

    # ConvTranspose formula: out = (in-1)*stride - 2*pad + dilation*(kernel-1) + output_padding + 1
    #          => output_padding = out - (in-1)*stride + 2*pad - dilation*(kernel-1) - 1
    paddings = NTuple{2, Int}[ ]
    for i in n_layers:-1:1
        size_in = sizes[i+1]
        size_out = sizes[i]
        push!(paddings,
            size_out .- (dilation .* (kernel .- Ref(1))) .+
            (pad .* Ref(2)) .- (stride .* (size_in .- Ref(1))) .- Ref(1)
        )
    end

    return paddings
end

"""
Defines the first half of the neural net,
   which compresses the input mel-spectrogram into a small number of visualizable outputs.

To train this half without labeled data you also need a 'decoder' afterwards
   which unpacks the outputs into a reconstruction of the input.
"""
function nn_chain_encoder(settings::NN_Settings)
    convolved_size = nn_chain_shape_after_convolution(settings)
    return Chain(
        # Convolutional layer with 32 different 3x3 kernels,
        #    offsetting each kernel application by 2 pixels to halve the output resolution vs input.
        Conv((3, 3), 1 => 32, relu;
             stride=2, pad=1),
        # Convolutional layer with 64 different 3x3 kernels,
        #    offsetting each kernel applicationo by 2 pixels to halve the output resolution again.
        Conv((3, 3), 32 => 64, relu;
             stride=2, pad=1),

        # Flatten the convolutional features into a 1D array (plus batch axis), for traditional NN layers:
        x -> reshape(x, :, size(x, 4)),

        # Add a traditional DNN layer.
        Dense(prod(convolved_size) => 128, relu),

        # Add the output layer.
        Dense(128 => settings.n_outputs)
    )
end

"""
Defines the second half of the neural net,
   which decompresses the visualizer output into the input (a mel-spectrogram).

This half is needed to train the first half without labeled data.
"""
function nn_chain_decoder(settings::NN_Settings)
    input_size = nn_input_size(settings.input_settings, settings.chunk_size_samples)
    convolved_size = nn_chain_shape_after_convolution(settings)
    conv_paddings = nn_convolution_layer_padding(input_size, 2,
                                                 (2, 2), (1, 1), (3, 3), (1, 1))
    return Chain(
        # Invert the layers from the encoder.
        Dense(settings.n_outputs => 128, relu),
        Dense(128 => prod(convolved_size), relu),
        x -> reshape(x, (convolved_size..., size(x, 2))),
        ConvTranspose((3, 3), 64 => 32, relu;
                      stride=2, pad=1,
                      outpad=conv_paddings[1]),
        ConvTranspose((3, 3), 32 => 1;
                      stride=2, pad=1,
                      outpad=conv_paddings[2])
    )
end

nn_chain_full(settings::NN_Settings) = Chain(nn_chain_encoder(settings), nn_chain_decoder(settings))


export NN_Settings, nn_chain_encoder, nn_chain_decoder, nn_chain_full