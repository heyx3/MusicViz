
function grab_mono_from_wav(file_path, channel=1)
    data = wavread(file_path)
    return (data[1][:, channel], data[2])
end

"Grab a slice of the given wav file samples, in seconds"
function sample_subset(samples, sample_rate, start_seconds, duration_seconds)
    @bp_check(ndims(samples) == 1, size(samples))
    first_sample = clamp(1 + Int(round(start_seconds * sample_rate)),
                         1, length(samples))
    last_sample = clamp(Int(round(first_sample + (duration_seconds * sample_rate))),
                        first_sample, length(samples))
    return @view samples[first_sample:last_sample]
end

function plot_audio_samples(samples, sample_rate)
    plot((1:length(samples))/sample_rate,
         samples)
end
function plot_audio_spectrogram(samples, sample_rate, start_seconds=0)
    S = spectrogram(samples,
                    Int(round(sample_rate/40)),
                    Int(round(sample_rate/100));
                    window=hanning)

    t = time(S)
    f = freq(S)
    p = reverse(log10.(power(S)))
    extent = [
        start_seconds + first(t)/sample_rate,
          start_seconds + last(t)/sample_rate,
        sample_rate/1000 * first(f),
          sample_rate/1000 * last(f)
    ]

    imshow(p, extent=extent, aspect="auto")
    xlabel("Timestamp (s)")
    ylabel("Frequency (KHz)")
    return S
end

export grab_mono_from_wav, sample_subset,
       plot_audio_samples, plot_audio_spectrogram