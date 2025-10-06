
function grab_mono_from_wav(file_path, channel=1)
    data = wavread(file_path)
    return (convert.(Ref(Float32), data[1][:, channel]), data[2])
end

function sample_range(sample_rate, start_seconds, duration_seconds)
    first_sample = 1 + Int(round(start_seconds * sample_rate))
    last_sample = first_sample + # NOTE: first_sample deliberately not added *inside* the Int(round(...)),
                                 #   for deterinism purposes
                     Int(round((duration_seconds * sample_rate)))
    return first_sample:last_sample
end

function samples_with_padding(all_samples::AbstractVector, samples_range::UnitRange{<:Integer})
    @bp_check(length(all_samples) > 0, "Didn't code this to handle empty sample arrays")

    front_padding = -min(first(samples_range), 0)
    back_padding = max(0, last(samples_range) + front_padding - length(all_samples))

    return PaddedView(0.0f0, all_samples,
                      (length(all_samples) + front_padding + back_padding, ),
                      (1 + front_padding, ))
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

export grab_mono_from_wav, sample_range, samples_with_padding,
       plot_audio_samples, plot_audio_spectrogram