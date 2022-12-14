clear all;
close all;
clc;

args = argv();
video_name = args{1};

wav_path = ["/wiset/Input/" video_name "/" video_name ".wav"];
output_path = ["/wiset/Output/SSSL/" video_name "/" video_name "_saliency.mat"];

disp(wav_path)


infoaudio = audioinfo(wav_path);
sdata = audioread(wav_path);
t = linspace(0, infoaudio.Duration, length(sdata));
Fs = infoaudio.SampleRate;
audio_in_seconds = zeros(int16(infoaudio.Duration), Fs, 6);
disp(Fs);

for i = 1:infoaudio.Duration;
    currentClip = reshape(sdata((i-1)*Fs+1: i*Fs, :), Fs, 4);
    W = currentClip(:, 1);
    ch2 = currentClip(:, 2);
    ch3 = currentClip(:, 3);
    ch4 = currentClip(:, 4);
        
    p2 = (sqrt(2)*W + ch2) / 2;
    p3 = (sqrt(2)*W + ch3) / 2;
    p4 = (sqrt(2)*W + ch4) / 2;

    n2 = (sqrt(2)*W - ch2) / 2;
    n3 = (sqrt(2)*W - ch3) / 2;
    n4 = (sqrt(2)*W - ch4) / 2;
        
    sp2 = spectrum1DwithMelCepstrumTrial(p2, infoaudio,1)';
    sp3 = spectrum1DwithMelCepstrumTrial(p3, infoaudio,1)';
    sp4 = spectrum1DwithMelCepstrumTrial(p4, infoaudio,1)';
        
    sn2 = spectrum1DwithMelCepstrumTrial(n2, infoaudio,1)';
    sn3 = spectrum1DwithMelCepstrumTrial(n3, infoaudio,1)';
    sn4 = spectrum1DwithMelCepstrumTrial(n4, infoaudio,1)';
        
    windowSize = infoaudio.SampleRate/2;
    b = (1/windowSize)*ones(1,windowSize);
    a = 1;

    hp2 = filter(b, a, sp2);
    hp3 = filter(b, a, sp3);
    hp4 = filter(b, a, sp4);
        
    hn2 = filter(b, a, sn2);
    hn3 = filter(b, a, sn3);
    hn4 = filter(b, a, sn4);

    audio_in_seconds(i, :, 1) = hp2;
    audio_in_seconds(i, :, 2) = hn2;
    audio_in_seconds(i, :, 3) = hp3;
    audio_in_seconds(i, :, 4) = hn3;
    audio_in_seconds(i, :, 5) = hp4;
    audio_in_seconds(i, :, 6) = hn4;

endfor

save(output_path, 'audio_in_seconds', '-v6');
