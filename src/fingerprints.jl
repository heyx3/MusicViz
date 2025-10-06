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
Base.print(io::IO, fp::Fingerprint) = print(io, "{ ",
    "str=", @sprintf("%.3f", fp.strength_t), ", ",
    "chaos=", @sprintf("%.3f", fp.chaos_t), ", ",
    "bright=", @sprintf("%.3f", fp.brightness_t), ", ",
    "pure=", @sprintf("%.3f", fp.purity_t), ", ",
    "roll=", @sprintf("%.1f", fp.rolloff_point_hz),
" }")

function Fingerprint(sample_rate::Float32,
                     samples::AbstractVector{Float32},
                     fft_samples::AbstractVector{Float32},
                     signal_noise_epsilon::Float32 = @f32(0.0001),
                     ;
                     buffer1::Vector{Float32} = Float32[ ])
    n_samples::Int = length(samples)
    @bp_check(n_samples > 0, "No buffer samples!")

    n_fft_samples::Int = length(fft_samples)
    @bp_check(n_fft_samples == (n_samples ÷ 2) + 1,
              "Wrong number of FFT samples! ",
                n_fft_samples, " instead of ", (n_samples ÷ 2) + 1)
    frequencies = (0 : n_fft_samples-1) * convert(Float32, sample_rate / n_samples)

    fft_samples_normalized = buffer1
    resize!(fft_samples_normalized, length(fft_samples))
    fft_samples_normalized .= fft_samples
    fft_samples_normalized ./= Ref(sum(fft_samples) + eps(Float32))

    rolloff_idx = find_first_of_iter(c >= 0.95 for c in iter_cumsum(fft_samples_normalized))
    @bp_check(exists(rolloff_idx))
    rolloff = frequencies[rolloff_idx]

    centroid = sum(a*b for (a,b) in zip(frequencies, fft_samples_normalized))
    zero_crossing_count = count(consecutive_pairs(samples)) do (a,b)
        return (abs(a - b) > signal_noise_epsilon) && (sign(a) == -sign(b))
    end
    return Fingerprint(
        sqrt(mean(s^2 for s in samples)),
        zero_crossing_count / max(1, (n_samples - 1)),
        centroid,
        sqrt(sum(((freq - centroid) ^ 2) * fft_samp_n for (freq, fft_samp_n) in zip(frequencies, fft_samples_normalized))),
        rolloff
    )
end

"Grabs the `Fingerprint` of a number of audio sections using the given FFT settings"
function get_fingerprints(samples::AbstractVector{Float32},
                          sample_rate::Float32,
                          snapshots::Vector{@NamedTuple{start_seconds::Float32,
                                                        duration_seconds::Float32}},
                          window = hanning
                          ;
                          buffer1::Vector{ComplexF32} = ComplexF32[ ],
                          buffer2::Vector{Float32} = Float32[ ],
                          buffer3::Vector{Float32} = Float32[ ]
                         )::Vector{Fingerprint}
    return map(snapshots) do snapshot
        # Get the sample subset.
        first_sample = max(1, 1 + Int(round(snapshot.start_seconds * sample_rate)))
        last_sample = min(length(samples),
                          first_sample + Int(round(snapshot.duration_seconds * sample_rate)))
        sample_view = @view samples[first_sample:last_sample]
        n_samples = length(sample_view)

        # Get the FFT.
        resize!(buffer1, n_samples)
        buffer1 .= sample_view .* window(n_samples)
        fft!(buffer1)

        # Copy the important values into a float buffer.
        resize!(buffer2, (n_samples ÷ 2) + 1)
        buffer2 .= abs.(@view buffer1[1:length(buffer2)])
        fft_samples = buffer2

        # Extract the fingerprint.
        return Fingerprint(sample_rate, sample_view, fft_samples; buffer1=buffer3)
    end
end

export Fingerprint, get_fingerprints