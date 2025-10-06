# Implementation of a neural-net approach to visualization.

"
Turns an audio sample into a matrix of continuous Float32 FFT results
    by moving a window across the samples.

Returns a matrix where each column is an FFT in one window,
   `window_size_samples x (length(samples) ÷ window_sample_overlap)`.

The output samples are not in Hz with amplitudes,
   but perceptually-linear Mels with perceptually-linear volume.
"
function nn_get_input_spectrogram(samples::AbstractVector{<:AbstractFloat},
                                  sample_rate::Float32,
                                  window = hanning
                                  ;
                                  window_size_samples::Int = 1024,
                                  window_sample_overlap::Int = window_size_samples ÷ 2,
                                  n_frequency_buckets::Int = 64,
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
    stft_samples_full = DSP.stft(buffer, window_size_samples, window_sample_overlap,
                                 window=window, fs=sample_rate)
    stft_samples_mag = abs.(stft_samples_full)

    # Convert FFT frequencies from Hz (physically-linear) to Mels (perceptually-linear).
    (n_stft_bins, n_frames) = size(stft_samples_mag)
    # stft_frequencies = range(start=0.0f0,
    #                          stop=sample_rate/2.0f0,
    #                          length=Float32(n_stft_bins))
    mel_samples = transpose(melscale_filterbanks(n_freqs = n_stft_bins,
                                                 n_mels = n_frequency_buckets,
                                                 sample_rate = Int(sample_rate),
                                                 fmin=0.0f0,
                                                 fmax=sample_rate/2)
                           ) * stft_samples_mag
    mel_samples .= log.(mel_samples .+ @f32(1e-6))
end

export nn_get_input_spectrogram


"
A lazy-iterator over the neural network inputs for small chunks of time (e.g. 1 second),
   usually with overlap between the chunks.

All returned inputs are exactly the same size (`window_size_samples` x
   `Int(sample_rate * chunk_size_seconds) ÷ window_sample_overlap`).

Iteration re-uses allocations internally, so you shouldn't hold onto data from one chunk
   while moving to the next!
"
@kwdef struct nn_chunked_audio{TSamples<:AbstractVector{<:AbstractFloat}, TWindow}
    samples::TSamples
    sample_rate::Float32
    chunk_size_seconds::Float32
    chunk_shift_seconds::Float32

    # Inside each chunk the FFT is an STFT, a.k.a. a moving window of the chunk's samples.
    window::TWindow = hanning
    window_size_samples::Int = 1024
    window_sample_overlap::Int = 512
    n_frequency_buckets::Int = 64

end
nn_chunked_audio(samples, sample_rate, chunk_size_seconds, chunk_shift_seconds; kw...) =
    nn_chunked_audio(; samples=samples, sample_rate=sample_rate,
                       chunk_size_seconds=chunk_size_seconds,
                       chunk_shift_seconds=chunk_shift_seconds)

Base.IteratorSize(::Type{<:nn_chunked_audio}) = Base.SizeUnknown()
Base.eltype(::Type{<:nn_chunked_audio}) = Matrix{Float32}

function Base.iterate(chunker::nn_chunked_audio)
    if isempty(chunker.samples)
        return nothing
    end

    return impl_chunked_audio(
        chunker,
        sample_range(chunker.sample_rate, 0.0f0, chunker.chunk_size_seconds),
        0.0f0,
        Float32[ ]
    )
end
function Base.iterate(chunker::nn_chunked_audio, (prev_t, buffers...))
    next_t = prev_t + chunker.chunk_shift_seconds
    next_sample_idcs = sample_range(
        chunker.sample_rate,
        next_t,
        chunker.chunk_size_seconds
    )

    if first(next_sample_idcs) > length(chunker.samples)
        return nothing
    else
        return impl_chunked_audio(chunker, next_sample_idcs, next_t, buffers...)
    end
end

function impl_chunked_audio(chunker::nn_chunked_audio,
                            sample_range::UnitRange{<:Integer}, sample_start_t::Float32,
                            buffer1::Vector{Float32})
    pseudo_spectrogram = nn_get_input_spectrogram(
        samples_with_padding(chunker.samples, sample_range),
        chunker.sample_rate,
        chunker.window,

        window_size_samples=chunker.window_size_samples,
        window_sample_overlap=chunker.window_sample_overlap,
        n_frequency_buckets=chunker.n_frequency_buckets,

        buffer = buffer1
    )
    return (pseudo_spectrogram, (sample_start_t, buffer1))
end

export nn_chunked_audio