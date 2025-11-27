"use client";

import { useEffect, useState } from "react";
import { PDFViewer } from "./components/PDFViewer";

// import { toast } from "react-hot-toast";

export default function Home() {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hide, setHide] = useState(false);
  const [showPdf, setShowPdf] = useState(false);
  const [pdfFile, setPdfFile] = useState('/pdflab.pdf');

  useEffect(() => {
    const handleChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };

    document.addEventListener("fullscreenchange", handleChange);

    return () => {
      document.removeEventListener("fullscreenchange", handleChange);
    };
  }, []);

  const toggleFullscreen = async () => {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.error("Failed to toggle fullscreen", err);
    }
  };
  const pulseShapingstr: string = `
  fs = 1e6;  % Sampling frequency (Hz)   
fc = 2.4e9;  % Carrier frequency (Hz)   
v = 30;  % Mobile speed (m/s)   
d = 100;  % Distance between transmitter and receiver (m)   
N = 1000;  % Number of samples   
% Generate random data   
data = randi([0 1], N, 1);   
% Modulate data using QPSK   
mod_data = qpsk_mod(data, fc, fs);   
% Add fading effects using Rayleigh fading channel   
h = rayleigh_chan(fc, fs, v, d, N);   
faded_data = mod_data .* h;   
% Add Doppler effects using Doppler shift   
fd = doppler_shift(fc, v, d);   
doppler_data = faded_data .* exp(1j * 2 * pi * fd * (0:N-1)'/fs);   
% Add noise   
noise = randn(N, 1) + 1j * randn(N, 1);   
received_data = doppler_data + noise;   
% Demodulate received data using QPSK   
demod_data = qpsk_demod(received_data, fc, fs);   
% Plot results   
figure;   
subplot(2, 1, 1);   
plot(abs(mod_data));   
title('Transmitted Signal');   
xlabel('Time (s)');   
ylabel('Amplitude');   
subplot(2, 1, 2);   
plot(abs(received_data));   
title('Received Signal');   
xlabel('Time (s)');   
ylabel('Amplitude');   
% Functions   
function mod_data = qpsk_mod(data, fc, fs)   
% QPSK modulation   
mod_data = zeros(size(data));   
for i = 1:length(data) 
if data(i) == 0   
mod_data(i) = exp(1j * pi/4);   
else   
mod_data(i) = exp(1j * 3*pi/4);   
end   
end   
mod_data = mod_data .* exp(1j * 2 * pi * fc * (0:length(data)-1)'/fs);   
end   
   
function demod_data = qpsk_demod(data, fc, fs)   
   % QPSK demodulation   
   demod_data = zeros(size(data));   
   for i = 1:length(data)   
      if angle(data(i)) < pi/4   
        demod_data(i) = 0;   
      else   
        demod_data(i) = 1;   
      end   
   end   
end   
   
function h = rayleigh_chan(fc, fs, v, d, N)   
   % Rayleigh fading channel   
   h = zeros(N, 1);   
   for i = 1:N   
      h(i) = sqrt(1/2) * (randn + 1j * randn);   
   end   
   h = h .* exp(-1j * 2 * pi * fc * d / (3e8));   
end   
   
function fd = doppler_shift(fc, v, d)   
   % Doppler shift   
   fd = v * fc / (3e8);   
end`
 const fadingstr: string = `
 sampleRate500KHz = 500e3;    
sampleRate20KHz  = 20e3;     
% Sample rate of 500 KHz 
% Sample rate of 20 KHz 
maxDopplerShift  = 200; % Max Doppler shift of diffuse components (Hz) 
delayVector = (0:5:15)*1e-6; % Discrete delays of four-path channel (s) 
gainVector  = [0 -3 -6 -9];  % Average path gains (dB) 
KFactor = 10;           
% Linear ratio of specular to diffuse power 
specDopplerShift = 100; % Doppler shift of specular component (Hz) 
rayChan = comm.RayleighChannel( ... 
SampleRate=sampleRate500KHz, ... 
PathDelays=delayVector, ...ss 
AveragePathGains=gainVector, ... 
MaximumDopplerShift=maxDopplerShift, ... 
RandomStream="mt19937ar with seed", ... 
Seed=10, ... 
PathGainsOutputPort=true); 
ricChan = comm.RicianChannel( ... 
SampleRate=sampleRate500KHz, ... 
PathDelays=delayVector, ... 
AveragePathGains=gainVector, ... 
KFactor=KFactor, ... 
DirectPathDopplerShift=specDopplerShift, ... 
MaximumDopplerShift=maxDopplerShift, ... 
RandomStream="mt19937ar with seed", ... 
Seed=100, ... 
PathGainsOutputPort=true); 
M = 4;               
% QPSK modulation 
phaseoffset = pi/4;  % Phase offset for QPSK 
bitsPerFrame = 1000; 
msg = randi([0 1],bitsPerFrame,1); 
%%  
modSignal = pskmod(msg,M,phaseoffset,InputType='bit'); 
rayChan(modSignal); 
ricChan(modSignal); 
release(rayChan); 
release(ricChan); 
figure(1); 
rayChan.Visualization = "Impulse and frequency responses"; 
rayChan.SamplesToDisplay = "100%"; 
% Display impulse and frequency responses for 2 frames 
numFrames = 2; 
for i = 1:numFrames 
% Create random data 
msg = randi([0 1],bitsPerFrame,1); 
% Modulate data 
modSignal = pskmod(msg,M,phaseoffset,InputType='bit'); 
% Filter data through channel and show channel responses 
rayChan(modSignal); 
end 
release(rayChan); 
rayChan.Visualization = "Doppler spectrum"; 
% Display Doppler spectrum from 5000 frame transmission 
numFrames = 5000; 
for i = 1:numFrames 
msg = randi([0 1],bitsPerFrame,1); 
modSignal = pskmod(msg,M,phaseoffset,InputType='bit'); 
rayChan(modSignal); 
end 
release(rayChan); 
rayChan.Visualization = "Impulse and frequency responses"; 
rayChan.SampleRate = sampleRate20KHz; 
rayChan.SamplesToDisplay = "25%";  % Display one of every four samples 
% Display impulse and frequency responses for 2 frames 
numFrames = 2; 
for i = 1:numFrames 
msg = randi([0 1],bitsPerFrame,1); 
modSignal = pskmod(msg,M,phaseoffset,InputType='bit'); 
rayChan(modSignal); 
end 
release(rayChan); 
rayChan.PathDelays = 0;        
% Single fading path with zero delay 
rayChan.AveragePathGains = 0;  % Average path gain of 1 (0 dB) 
for i = 1:numFrames 
msg = randi([0 1],bitsPerFrame,1); 
modSignal = pskmod(msg,M,phaseoffset,InputType='bit'); 
rayChan(modSignal); 
end 
release(rayChan); 
rayChan.Visualization = "Off"; % Turn off Rayliegh object visualization 
ricChan.Visualization = "Off"; % Turn off Rician object visualization 
% Same sample rate and delay profile for the Rayleigh and Rician objects 
ricChan.SampleRate = rayChan.SampleRate; 
ricChan.PathDelays = rayChan.PathDelays; 
ricChan.AveragePathGains = rayChan.AveragePathGains; 
% Configure a Time Scope System object to show path gain magnitude 
gainScope = timescope( ... 
SampleRate=rayChan.SampleRate, ... 
TimeSpanSource="Property",... 
TimeSpan=bitsPerFrame/2/rayChan.SampleRate, ... % One frame span 
Name="Multipath Gain", ... 
ChannelName=["Rayleigh","Rician"], ... 
ShowGrid=true, ... 
YLimits=[-40 10], ... 
YLabel="Gain (dB)"); 
% Compare the path gain outputs from both objects for one frame 
msg = randi([0 1],bitsPerFrame,1); 
modSignal = pskmod(msg,M,phaseoffset,InputType='bit'); 
[~,rayPathGain] = rayChan(modSignal); 
[~,ricPathGain] = ricChan(modSignal); 
% Form the path gains as a two-channel input to the time scope 
gainScope(10*log10(abs([rayPathGain,ricPathGain]).^2));
`
const signalstr:string = `
2. SIGNAL TRANSMISSION AND RECEPTION: 
M = 4; % QPSK signal constellation 
no_of_data_points = 64; % have 64 data points 
block_size = 8; % size of each ofdm block 
cp_len = ceil(0.1*block_size); % length of cyclic prefix 
no_of_ifft_points = block_size; % 8 points for the FFT/IFFT 
no_of_fft_points = block_size; 
% 1. Generate 1 x 64 vector of random data points 
data_source = randsrc(1, no_of_data_points, 0:M-1); 
figure(1) 
stem(data_source); grid on; xlabel('Data Points'); ylabel('Amplitude') 
title('Transmitted Data "O"') 
% 2. Perform QPSK modulation 
qpsk_modulated_data = pskmod(data_source, M); 
scatterplot(qpsk_modulated_data);title('MODULATED TRANSMITTED DATA'); 
% 3. Do IFFT on each block 
% Make the serial stream a matrix where each column represents a pre-OFDM 
% block (w/o cyclic prefixing) 
% First: Find out the number of colums that will exist after reshaping 
num_cols=length(qpsk_modulated_data)/block_size; 
data_matrix = reshape(qpsk_modulated_data, block_size, num_cols); 
% Second: Create empty matix to put the IFFT'd data 
cp_start = block_size-cp_len; 
cp_end = block_size; 
% Third: Operate columnwise & do CP 
for i=1:num_cols, 
ifft_data_matrix(:,i) = ifft((data_matrix(:,i)),no_of_ifft_points); 
% Compute and append Cyclic Prefix 
for j=1:cp_len 
actual_cp(j,i) = ifft_data_matrix(j+cp_start,i); 
end 
% Append the CP to the existing block to create the actual OFDM block 
ifft_data(:,i) = vertcat(actual_cp(:,i),ifft_data_matrix(:,i)); 
end 
% 4. Convert to serial stream for transmission 
[rows_ifft_data cols_ifft_data]=size(ifft_data); 
len_ofdm_data = rows_ifft_data*cols_ifft_data; 
% Actual OFDM signal to be transmitted 
ofdm_signal = reshape(ifft_data, 1, len_ofdm_data); 
figure(3) 
plot(real(ofdm_signal)); xlabel('Time'); ylabel('Amplitude'); 
title('OFDM Signal');grid on; 
% ------------------------------------------ 
% C: % +++++ HPA +++++ 
% ------------------------------------------ 
%To show the effect of the PA simply we will add random complex noise 
%when the power exceeds the avg. value, otherwise it add nothing. 
% 1. Generate random complex noise 
noise = randn(1,len_ofdm_data) + sqrt(-1)*randn(1,len_ofdm_data); 
% 2. Transmitted OFDM signal after passing through HPA 
avg=0.4; 
for i=1:length(ofdm_signal) 
if ofdm_signal(i) > avg 
ofdm_signal(i) = ofdm_signal(i)+noise(i); 
end 
if ofdm_signal(i) < -avg 
ofdm_signal(i) = ofdm_signal(i)+noise(i); 
end 
end 
figure(4) 
plot(real(ofdm_signal)); xlabel('Time'); ylabel('Amplitude'); 
title('OFDM Signal after HPA');grid on; 
channel=randn(1,block_size)+sqrt(-1)*randn(1,block_size); 
% 1. Pass the ofdm signal through the channel 
after_channel = filter(channel, 1, ofdm_signal); 
% 2. Add Noise 
awgn_noise = awgn(zeros(1,length(after_channel)),0); 
% 3. Add noise to signal... 
recvd_signal = awgn_noise+after_channel; 
% 4. Convert Data back to "parallel" form to perform FFT 
recvd_signal_matrix = reshape(recvd_signal,rows_ifft_data, cols_ifft_data); 
% 5. Remove CP 
recvd_signal_matrix(1:cp_len,:)=[]; 
% 6. Perform FFT 
for i=1:cols_ifft_data 
% FFT 
fft_data_matrix(:,i) = fft(recvd_signal_matrix(:,i),no_of_fft_points); 
end 
% 7. Convert to serial stream 
recvd_serial_data = reshape(fft_data_matrix, 1,(block_size*num_cols)); 
scatterplot(recvd_serial_data);title('MODULATED RECEIVED DATA'); 
% 8. Demodulate the data 
qpsk_demodulated_data = pskdemod(recvd_serial_data,M); 
scatterplot(recvd_serial_data);title('MODULATED RECEIVED DATA'); 
figure; 
stem(qpsk_demodulated_data,'rx'); 
grid on;xlabel('Data Points');ylabel('Amplitude');title('Received Data "X"')`

  const syncstr = `
  enb.NDLRB = 15; % Number of resource blocks 
enb.CellRefP1:% One transmit antenna port 
enb.NCellID=10; % Cell ID 
enb.CyclicPrefix= 'Normal'; % Normal cyclic prefix 
enb.DuplexMode = 'FDD'; % FDD 
SNRdB=22;% Desired SNR in dB 
SNR=10^(SNRdB/20); % Linear SNR 
mg('default'); % Configure random number generators 
cfg.Seed=1; % Channel seed 
cfg.NRxAnts = 1; % 1 receive antenna 
cfg. DelayProfile = 'EVA'; % EVA delay spread 
cfg DopplerFreq=120:% 120Hz Doppler frequency 
cfg.MIMOCorrelation 'Low; % Low (no) MIMO correlation. 
cfg.InitTime = 0; % Initialize at time zero 
cfg.NTerms 16, % Oscillators used in fading model 
cfg.ModelType 'GMEDS'; % Rayleigh fading model type 
cfg.InitPhase 'Random'; % Random initial phases 
cfg.NormalizePathGains = 'On'; % Normalize delay profile power 
cfg.NormalizeTxAnta On'; % Normalize for transmit antennas 
cec.PilotAverage = "UserDefined'; % Pilot averaging method 
cec.Freq Window = 9; % Frequency averaging window in REs 
cec.TimeWindow=9; % Time averaging window in REs gridsize = 
IteDLResourceGridSize(enb); 
K=gridaize(1); % Number of subearriers 
L=gridsize(2); % Number of OFDM symbols in 
one subframeP= gridsize(3); % Number of transmit antenna ports  
txGrid=[]; 
%Number of bits needed is size of resource grid (K*L*P) number of hits per symbol (2 for QPSK) 
numberOfBits=K*L*P*2; 
% Create random bit stream 
inputBits =randi([0 1], numberOfBits, 1); 
% Modulate input bits 
inputSym= IteSymbolModulate(inputBits, 'QPSK'); 
% For all subframes within the framefor 
sf = 0:10 
% Set subframe number 
enb.NSubframe = mod(sf,10); 
% Generate empty subframe subframe = 
IteDLResourceGrid(enb); 
% Map input symbols to grid 
subframe(:) = inputSym; 
% Generate synchronizing signals 
pssSym= ltePSS(enb); 
sssSym =IteSSS(enb);  
pssInd= ItePSSIndices(enb); 
sssInd = IteSSSIndices(enb); 
% Map synchronizing signals to the grid 
subframe(pssInd) = pssSym; 
subframe(sasind) = sssSym; 
%Generate cell specific reference 
signal symbols and indices cellRsSym =lteCellRS(enb); 
cellRsInd = IteCellRSIndices(enb); 
% Map cell specific reference signal to grid 
subframe(cellRsInd) = cellRsSym; 
% Append subframe to grid to be transmitted 
txGrid = [txGrid subframe]; %#ok 
end 
[txWaveform,info] = IteOFDMModulate(enb,txGrid); 
txGrid = txGrid(:,1:140); 
cfg.Sampling Rate = info.Sampling Rate; 
% Pass data through the fading channel model 
rx Waveform = IteFadingChannel(cfg,txWaveform); 
% Calculate noise gain 
NO=1/(sqrt(2.0*enb. CellRefP*double(info.Nfft))*SNR);`

  const copySync = ()=>{
    navigator.clipboard
    .writeText(syncstr)
    .then(()=>{
    })
    .catch((err)=>{
    });
  }

  const copyPulseShaping = () => {
    navigator.clipboard
      .writeText(pulseShapingstr)
      .then(() => {
        // toast.success("Pulse Shaping copied to clipboard");
      })
      .catch((err) => {
        console.error("Failed to copy Pulse Shaping", err);
        // toast.error("Failed to copy Pulse Shaping to clipboard");
      });
  };

  const copyFading = () => {
    navigator.clipboard
      .writeText(fadingstr)
      .then(() => {
        // toast.success("Fading copied to clipboard");
      })
      .catch((err) => {
        console.error("Failed to copy Fading", err);
        // toast.error("Failed to copy Fading to clipboard");
      });
  };

  const copySignal = () => {
    navigator.clipboard
      .writeText(signalstr)
      .then(() => {
        // toast.success("Signal copied to clipboard");
      })
      .catch((err) => {
        console.error("Failed to copy Signal", err);
        // toast.error("Failed to copy Signal to clipboard");
      });
  };

  const toggleContentVisibility = () => {
    setHide((prev) => !prev);
  };
  return (
    <main className="relative h-screen overflow-hidden">
      {/* Background image fills the entire window */}
      <img
        src="/image.jpg"
        alt=""
        aria-hidden="true"
        className="fixed inset-0 w-full h-full object-cover"
      />

      {/* Fullscreen toggle button (spaced slightly from edges) */}
      <div className="absolute z-20 top-6 right-6 flex flex-col gap-3 items-end">
        <button
          type="button"
          onClick={toggleFullscreen}
          className="rounded-md bg-black/60 px-4 py-2 text-sm font-medium text-white hover:bg-black/80 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/80"
        >
          {isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
        </button>
        {/* Button to toggle showing/hiding the copy buttons */}
        <button
          type="button"
          onClick={toggleContentVisibility}
          className="rounded-md bg-black/60 px-3 py-1.5 text-xs font-medium text-white hover:bg-black/80 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/80"
        >
          {hide ? "Hide controls" : "Show controls"}
        </button>
      </div>

      {/* Content wrapper stays above the background */}
      {hide && (
        <div className="relative z-10 flex items-center justify-center h-full">
          {/* Buttons are centered with spacing */}
          <div className="flex flex-col items-center gap-4 px-4">
            <button
              onClick={copyPulseShaping}
              className="min-w-[220px] rounded-md bg-white/80 px-4 py-2 text-sm font-medium text-black shadow hover:bg-white"
            >
              Copy Pulse Shaping
            </button>
            <button
              onClick={copyFading}
              className="min-w-[220px] rounded-md bg-white/80 px-4 py-2 text-sm font-medium text-black shadow hover:bg-white"
            >
              Copy Fading
            </button>
            <button
              onClick={copySignal}
              className="min-w-[220px] rounded-md bg-white/80 px-4 py-2 text-sm font-medium text-black shadow hover:bg-white"
            >
              Copy Signal Transmission and Reception
            </button>
            <button
              onClick={copySync}
              className="min-w-[220px] rounded-md bg-white/80 px-4 py-2 text-sm font-medium text-black shadow hover:bg-white"
            >
              Copy Synchronization
            </button>
            <div className="flex gap-2 items-center">
              <select
                value={pdfFile}
                onChange={(e) => setPdfFile(e.target.value)}
                className="min-w-[150px] rounded-md bg-white/80 px-3 py-2 text-sm font-medium text-black shadow"
              >
                <option value="/pdflab.pdf">pdflab.pdf</option>
                <option value="/RF%20lab.pdf">RF lab.pdf</option>
              </select>
              <button
                onClick={() => { setShowPdf(true); }}
                className="min-w-[100px] rounded-md bg-white/80 px-4 py-2 text-sm font-medium text-black shadow hover:bg-white"
              >
                Open PDF
              </button>
            </div>
            
          </div>
        </div>
      )}

      {/* PDF Viewer Modal */}
      {showPdf && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
          <div className="bg-white w-full h-full max-w-5xl max-h-screen rounded-lg shadow-lg flex flex-col">
            <PDFViewer file={pdfFile} onClose={() => setShowPdf(false)} />
          </div>
        </div>
      )}
    </main>
  );
}
