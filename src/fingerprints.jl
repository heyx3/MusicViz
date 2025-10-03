"The quantitative fingerprint of a specific sample+fft of audio"
struct Fingerprint
    # Root Mean Square (RMS), a.k.a. the volume, normalized to 0-1 range
    strength_t::Float32
    # Zero Crossing Rate (ZCR), indicates the noisiness, normalized to 0-1 range.
    chaos_t::Float32
    # The perceptual brightness of the timbre, normalized to 0-1 range
    brightness_t::Float32
    # The simplicity of the timbre; normalized to 0-1 range
    purity_t::Float32
    # Cutoff frequency below which most of the energy lies, in Hz
    rolloff_point_hz::Float32
end

function Fingerprint(sample_rate::Float32,
                     samples::AbstractVector{Float32},
                     fft_samples::AbstractVector{Float32},
                     n_frequency_buckets::Int
                     ;
                     buffer1::Vector{Float32} = Float32[ ])
    n_samples::Int = length(samples)
    @bp_check(n_samples > 0)
    @bp_check(length(fft_samples) == (n_samples ÷ 2) + 1,
              length(fft_samples), " instead of ", (n_samples ÷ 2) + 1)
    frequencies = (0 : n_samples-1) * convert(Float32, sample_rate / n_samples)

    fft_samples_normalized = buffer1
    resize!(fft_samples_normalized, length(fft_samples))
    fft_samples_normalized .= samples
    fft_samples_normalized ./= (sum(samples) + eps(Float32))

    rolloff_idx = findfirst(c >= 0.95 for c in iter_cumsum(fft_samples_normalized))
    @bp_check(exists(rolloff_idx))
    rolloff = frequencies[rolloff_idx]

    return Fingerprint(
        sqrt(mean(s^2 for s in samples)),
        sum(iter_diff(signbit(s) for s in samples))
          / length(samples),
        sum(a*b for (a,b) in zip(frequencies, fft_samples_normalized)),
        sqrt(sum(((freq - centroid) ^ 2) * fft_samp_n for (freq, fft_samp_n) in zip(frequencies, fft_samples_normalized))),
        rolloff
    )
end